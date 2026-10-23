# Experimental peeling features (branch exp/peeling-experiments).
#
#   1. Near-field RFI sources: a port of CPU TTCal's `RFISource` (the
#      ttcal_dev build used by the main pipeline for RFI peeling).
#   2. Tukey phase-prior regularization of the direction-dependent gains
#      (R. Byrne, "Regularized Peeling", 2026; CaliCo `cost_ddcal`).
#   3. A CPU-TTCal-compatible solver: RK4-integrated StefCal steps with
#      per-channel convergence, as in TTCal's `solve!`.
#
# Nothing here changes the default code paths: `gpu_stefcal!` and the
# existing kernels are untouched, and the new behaviour is only reached
# through `gpu_stefcal_exp!` / `GPURFISource`.

using LinearAlgebra

_mean(x) = sum(x) / length(x)
_median(x) = (v = sort(vec(x)); n = length(v); isodd(n) ? v[(n + 1) ÷ 2] : (v[n ÷ 2] + v[n ÷ 2 + 1]) / 2)

const SPEED_OF_LIGHT = 299792458.0

#==============================================================================#
#                         Near-field RFI sources                               #
#==============================================================================#

"""
Per-channel RFI spectrum.  Evaluation reproduces ttcal_dev's `RFISpectrum`
exactly: `searchsortedfirst` on the channel list, clamping outside it and
linear interpolation between the bracketing channels.  (The list is assumed
sorted; a mis-sorted model entry is reproduced, not repaired.)
"""
struct GPURFISpectrum
    channels::Vector{Float64}
    I::Vector{Float64}
    Q::Vector{Float64}
    U::Vector{Float64}
    V::Vector{Float64}
end

function (spectrum::GPURFISpectrum)(frequency::Float64)
    ch = spectrum.channels
    idx2 = searchsortedfirst(ch, frequency)
    idx2 == 1 && return (spectrum.I[1], spectrum.Q[1], spectrum.U[1], spectrum.V[1])
    N = length(ch)
    idx2 > N && return (spectrum.I[N], spectrum.Q[N], spectrum.U[N], spectrum.V[N])
    idx1 = idx2 - 1
    frequency == ch[idx1] && return (spectrum.I[idx1], spectrum.Q[idx1], spectrum.U[idx1], spectrum.V[idx1])
    frequency == ch[idx2] && return (spectrum.I[idx2], spectrum.Q[idx2], spectrum.U[idx2], spectrum.V[idx2])
    w = (frequency - ch[idx1]) / (ch[idx2] - ch[idx1])
    lerp(a) = a[idx1] + (a[idx2] - a[idx1]) * w
    return (lerp(spectrum.I), lerp(spectrum.Q), lerp(spectrum.U), lerp(spectrum.V))
end

"""Terrestrial (near-field) RFI emitter at a fixed ITRF position."""
struct GPURFISource <: GPUSource
    name::String
    position_itrf::SVector{3, Float64}
    spectrum::GPURFISpectrum
end

is_rfi(::GPUSource) = false
is_rfi(::GPURFISource) = true
is_rfi(s::AbstractGPUPeelingSource) = is_rfi(unwrap(s))

# Ground emitters are always peeled, as in CPU TTCal's isabovehorizon.
is_above_horizon(::GPURFISource, ::Float64, ::Float64, ::Float64) = true

"""WGS84 geodetic (deg, deg, m above ellipsoid) to ITRF XYZ (m)."""
function wgs84_to_itrf(lon_deg::Float64, lat_deg::Float64, height_m::Float64)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lon, lat = deg2rad(lon_deg), deg2rad(lat_deg)
    N = a / sqrt(1 - e2 * sin(lat)^2)
    SVector((N + height_m) * cos(lat) * cos(lon),
            (N + height_m) * cos(lat) * sin(lon),
            (N * (1 - e2) + height_m) * sin(lat))
end

"""Build a `GPURFISource` from a TTCal JSON entry (`rfi-frequencies`, `el`, ...)."""
function construct_gpu_rfi_source(c::Dict)
    name = get(c, "name", "")
    sys = get(c, "sys", "WGS84")
    el, lon, lat = Float64(c["el"]), Float64(c["long"]), Float64(c["lat"])
    # CPU TTCal: Position(pos"WGS84"/pos"ITRF", el, long, lat)
    position = sys == "WGS84" ? wgs84_to_itrf(lon, lat, el) :
               sys == "ITRF"  ? SVector(el, lon, lat) :
               error("RFI source '$name': unknown coordinate system $sys")
    channels = Float64.(c["rfi-frequencies"])
    N = length(channels)
    issorted(channels) || log_warning("RFI source '$name': rfi-frequencies are not sorted; " *
                                      "spectrum lookup reproduces CPU TTCal's behaviour")
    spectrum = GPURFISpectrum(channels, Float64.(c["rfi-I"]),
                              Float64.(get(c, "rfi-Q", zeros(N))),
                              Float64.(get(c, "rfi-U", zeros(N))),
                              Float64.(get(c, "rfi-V", zeros(N))))
    GPURFISource(name, position, spectrum)
end

"""
    uvw_geometry(meta) -> (s0, sign, rms)

Phase-centre unit vector ŝ₀ (ITRF) and the UVW sign convention of the MS,
from a least-squares fit of the MS `w` column to the ITRF antenna baselines:
w_α = sign · ŝ₀·(r₂ - r₁).  Snapshots are phased near zenith, so ŝ₀ is the
solution pointing up.  OVRO-LWA MSes have sign = -1 (UVW = r₁ - r₂), while
TTCalX's far-field kernels assume sign = +1; see `apply_uvw_convention!`.
`rms` is the fit residual (m).
"""
function uvw_geometry(meta::GPUMetadata)
    pos = Array(meta.antenna_positions)
    bl = Array(meta.baselines)
    w = Array(meta.uvw)[3, :]
    sel = findall(bl[1, :] .!= bl[2, :])
    B = permutedims(pos[:, bl[2, sel]] .- pos[:, bl[1, sel]])
    v = B \ w[sel]
    up = vec(sum(pos, dims=2)); up = up / norm(up)
    sign = dot(v, up) >= 0 ? 1 : -1
    s0 = SVector{3}(sign .* v ./ norm(v))
    rms = sqrt(sum(abs2, B * v .- w[sel]) / length(sel))
    s0, sign, rms
end

"""
    apply_uvw_convention!(meta) -> sign

Make `meta.uvw` follow the convention TTCalX's genvis kernels assume
(UVW = r₂ - r₁) by negating it when the MS stores r₁ - r₂.  Without this
the far-field model is the mirror image of the sky; free full-Jones gains
can still move it back, which hides the problem in plain peeling.
"""
function apply_uvw_convention!(meta::GPUMetadata)
    _, sign, _ = uvw_geometry(meta)
    sign < 0 && (meta.uvw .*= -1)
    sign
end

_rx(a) = @SMatrix [1.0 0.0 0.0; 0.0 cos(a) sin(a); 0.0 -sin(a) cos(a)]
_ry(a) = @SMatrix [cos(a) 0.0 -sin(a); 0.0 1.0 0.0; sin(a) 0.0 cos(a)]
_rz(a) = @SMatrix [cos(a) sin(a) 0.0; -sin(a) cos(a) 0.0; 0.0 0.0 1.0]

"""
    j2000_to_itrf_matrix(mjd_utc) -> R   (x_itrf = R * x_j2000)

IAU 1976 precession, the leading IAU 1980 nutation terms and GAST (UT1 ≈ UTC).
Polar motion, frame bias and aberration are ignored (≲ 20″), which is far
below what matters at OVRO-LWA resolution; precession since J2000 (~0.35°)
is not.  TTCalX's production genvis uses J2000 l,m about PHASE_DIR with the
MS UVW and so carries no precession/nutation at all.
"""
function j2000_to_itrf_matrix(mjd_utc::Float64)
    as = deg2rad(1 / 3600)
    jd = mjd_utc + 2400000.5
    T = (jd + 69.184 / 86400 - 2451545.0) / 36525
    ζ = (2306.2181T + 0.30188T^2 + 0.017998T^3) * as
    z = (2306.2181T + 1.09468T^2 + 0.018203T^3) * as
    θ = (2004.3109T - 0.42665T^2 - 0.041833T^3) * as
    P = _rz(-z) * _ry(θ) * _rz(-ζ)
    Ω = deg2rad(125.04452 - 1934.136261T)
    L = deg2rad(280.4665 + 36000.7698T)
    Lm = deg2rad(218.3165 + 481267.8813T)
    Δψ = (-17.20sin(Ω) - 1.32sin(2L) - 0.23sin(2Lm) + 0.21sin(2Ω)) * as
    Δε = (9.20cos(Ω) + 0.57cos(2L) + 0.10cos(2Lm) - 0.09cos(2Ω)) * as
    ε0 = (84381.448 - 46.8150T - 0.00059T^2 + 0.001813T^3) * as
    N = _rx(-(ε0 + Δε)) * _rz(-Δψ) * _rx(ε0)
    gmst = deg2rad(mod(280.46061837 + 360.98564736629 * (jd - 2451545.0) +
                       0.000387933T^2 - T^3 / 38710000, 360.0))
    _rz(gmst + Δψ * cos(ε0 + Δε)) * N * P
end

"""Unit vector of J2000 (ra, dec) in radians."""
_unit(ra, dec) = SVector(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec))

"""Mean TIME (MJD days, UTC) of an MS."""
function ms_mjd(ms_path::String)
    tables == PyNULL() && init_pycasacore()
    t = tables.table(ms_path, readonly=true, ack=false)
    tm = t.getcol("TIME")
    t.close()
    (minimum(tm) + maximum(tm)) / 2 / 86400
end

"""
    geometric_uvw!(meta, mjd_utc; sign=1) -> (rms_w, phase_centre_offset_deg)

Replace `meta.uvw` with ITRF baselines r₂ - r₁ rotated into J2000 and
projected on the (east, north, PHASE_DIR) basis, times `sign`.  With the
J2000 l,m,n TTCalX's kernels use, the far-field phase becomes the true
apparent geometry k·b·(ŝ - ŝ₀), as in CPU TTCal: verified on 2024-12-20
data, where it reproduces CPU TTCal's model (rel. diff 0.11 at 73 MHz, the
rest being aberration and the Gaussian shapes) and its correlation with
the data (0.81), while `sign = -1` gives the mirror image.  Returns the
rms difference from the MS `w` (with the MS sign) and the angle between the
rotated PHASE_DIR and the ŝ₀ fitted from the MS `w`, as sanity checks.
"""
function geometric_uvw!(meta::GPUMetadata, mjd_utc::Float64; sign::Int=1)
    R = j2000_to_itrf_matrix(mjd_utc)
    ra, dec = meta.phase_center_ra, meta.phase_center_dec
    ew = _unit(ra, dec)
    eu = SVector(-sin(ra), cos(ra), 0.0)
    ev = cross(ew, eu)
    E = permutedims(hcat(eu, ev, ew)) * R'   # rows: basis vectors in ITRF
    s0_fit, msign, _ = uvw_geometry(meta)
    offset = rad2deg(acos(clamp(dot(R * ew, s0_fit), -1.0, 1.0)))
    pos = Array(meta.antenna_positions)
    bl = Array(meta.baselines)
    w_ms = Array(meta.uvw)[3, :]
    uvw = zeros(3, size(bl, 2))
    for α in axes(bl, 2)
        b = SVector(pos[1, bl[2, α]] - pos[1, bl[1, α]], pos[2, bl[2, α]] - pos[2, bl[1, α]],
                    pos[3, bl[2, α]] - pos[3, bl[1, α]])
        uvw[:, α] = sign .* (E * b)
    end
    rms_w = sqrt(sum(abs2, sign .* uvw[3, :] .- msign .* w_ms) / size(bl, 2))
    copyto!(meta.uvw, uvw)
    rms_w, offset
end

"""
    gpu_genvis!(vis, meta, source::GPURFISource, ra, dec, lst; geometry=:uvw, position=nothing)

Add the near-field model of an RFI emitter, with CPU TTCal's geometry:

    τ_i = (D - |P - r_i| - ŝ₀·r_i) / c,   V_ij = S(ν) exp(2πiν(τ_i - τ_j))

with ITRF antenna positions r_i and emitter position P.  ŝ₀ comes from
`uvw_geometry` (`geometry=:uvw`); `geometry=:none` drops the ŝ₀ term, which
reproduces the feature/rfi-peeling-gpu branch.  `position` overrides P
(ITRF metres), e.g. to reproduce CPU TTCal's misread WGS84 position.
Stokes → linear uses TTCal's convention xy = U - iV, yx = U + iV.
"""
function gpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPURFISource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64;
                     geometry::Symbol=:uvw, position=nothing)
    positions = Array(meta.antenna_positions)
    channels = Array(meta.channels)
    baselines = Array(meta.baselines)
    Na, Nb, Nf = meta.Nant, meta.Nbase, meta.Nfreq
    P = position === nothing ? source.position_itrf : SVector{3}(position)
    # Per-antenna path: distance to the emitter (+ projection on ŝ₀).
    path = [norm(P - SVector(positions[1, i], positions[2, i], positions[3, i])) for i in 1:Na]
    if geometry == :uvw
        s0, _, _ = uvw_geometry(meta)
        for i in 1:Na
            path[i] += s0[1] * positions[1, i] + s0[2] * positions[2, i] + s0[3] * positions[3, i]
        end
    elseif geometry != :none
        error("unknown RFI geometry $geometry")
    end
    hxx = zeros(ComplexF64, Nb, Nf); hxy = similar(hxx); hyx = similar(hxx); hyy = similar(hxx)
    @inbounds for β in 1:Nf
        I, Q, U, V = source.spectrum(channels[β])
        fxx, fxy, fyx, fyy = complex(I + Q), complex(U, -V), complex(U, V), complex(I - Q)
        k = 2π * channels[β] / SPEED_OF_LIGHT
        for α in 1:Nb
            a1, a2 = baselines[1, α], baselines[2, α]
            fringe = cis(k * (path[a2] - path[a1]))
            hxx[α, β] = fxx * fringe; hxy[α, β] = fxy * fringe
            hyx[α, β] = fyx * fringe; hyy[α, β] = fyy * fringe
        end
    end
    if _is_gpu(vis)
        vis.xx .+= CuArray(hxx); vis.xy .+= CuArray(hxy)
        vis.yx .+= CuArray(hyx); vis.yy .+= CuArray(hyy)
    else
        vis.xx .+= hxx; vis.xy .+= hxy; vis.yx .+= hyx; vis.yy .+= hyy
    end
    vis
end

cpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPURFISource,
            ra::Float64, dec::Float64, lst::Float64; kw...) =
    gpu_genvis!(vis, meta, source, ra, dec, lst; kw...)

#==============================================================================#
#                     Tukey phase-prior regularization                          #
#==============================================================================#

"""
Per-gain phase prior from "Regularized Peeling": a source drift of Δθ shows
up as an antenna phase 2π sin(Δθ) (ν/c) |r_j|, so the prior on each
direction-dependent gain element (diagonal Jones terms) is

    P(φ) = 1                                      |φ| ≤ φ_start
         = ½ cos(π (|φ| - φ_start)/φ_width) + ½    φ_start < |φ| < φ_start + φ_width
         = 0                                      otherwise

with φ_start from `max_offset_deg` and φ_start + φ_width from
`max_offset_deg + taper_deg`.  Gains with φ_start ≥ π are left free.
|r_j| is the planar distance from the array centre (median antenna
position).  The cost minimised is Σ|V - GMGᴴ|² - λ Σ log P(arg g).

Note: the note's Eq. 9 and CaliCo's `tukey_taper` use 2π and the signed
phase inside the cosine, which makes the taper non-monotonic and asymmetric;
the intended π / |φ| form is used here.
"""
struct PhaseRegularization{T<:AbstractMatrix{Float64}}
    start::T      # (Nant, Nfreq) rad; Inf → unregularized
    width::T      # (Nant, Nfreq) rad
    lambda::Float64
    max_offset_deg::Float64
    taper_deg::Float64
end

"""Planar distance (m) of each antenna from the array centre (median position)."""
function antenna_planar_distances(positions::AbstractMatrix{Float64})
    center = [_median(positions[k, :]) for k in 1:3]
    up = center / norm(center)
    d = zeros(size(positions, 2))
    for i in axes(positions, 2)
        rel = positions[:, i] .- center
        d[i] = norm(rel .- dot(rel, up) .* up)
    end
    d
end

function phase_regularization(meta::GPUMetadata, max_offset_deg::Float64, taper_deg::Float64;
                              lambda::Float64=1.0, gpu::Bool=true, nfreq::Int=meta.Nfreq)
    dist = antenna_planar_distances(Array(meta.antenna_positions))
    channels = Array(meta.channels)
    nfreq == 1 && (channels = [_mean(channels)])
    k = 2π .* dist .* channels' ./ SPEED_OF_LIGHT
    start = k .* sin(deg2rad(max_offset_deg))
    stop = k .* sin(deg2rad(max_offset_deg + taper_deg))
    width = max.(stop .- start, 1e-12)
    start[start .>= π] .= Inf
    PhaseRegularization(gpu && CUDA.functional() ? CuArray(start) : start,
                        gpu && CUDA.functional() ? CuArray(width) : width,
                        lambda, max_offset_deg, taper_deg)
end

"""Fraction of (antenna, channel) gains that the prior can act on."""
regularized_fraction(reg::PhaseRegularization) = count(isfinite, Array(reg.start)) / length(reg.start)

@inline function tukey_prior(t::Float64, start::Float64, width::Float64)
    t <= start && return 1.0
    t >= start + width && return 0.0
    0.5 * cos(π * (t - start) / width) + 0.5
end

@inline function _reg_cost(t, a, r2s, start, width, lam)
    P = tukey_prior(t, start, width)
    P <= 0.0 && return Inf
    c = cos(t - a)
    data = c > 0.0 ? r2s * (1.0 - c * c) : r2s
    data - lam * log(P)
end

"""
Minimise s|z - z0|² - λ log P(arg z) over complex z (exact in the radius,
grid + golden-section in the phase).  Returns z0 when the prior is inactive.
"""
@inline function regularize_element(z0::ComplexF64, s::Float64, start::Float64,
                                    width::Float64, lam::Float64)
    (!isfinite(start) || lam <= 0.0 || !(s > 0.0)) && return z0
    φ0 = angle(z0)
    a = abs(φ0)
    a <= start && return z0
    r0 = abs(z0)
    r2s = s * r0 * r0
    hi = min(a, start + width)
    best_t = start
    best_f = _reg_cost(start, a, r2s, start, width, lam)
    N = 48
    @inbounds for k in 1:N
        t = start + (hi - start) * k / N
        f = _reg_cost(t, a, r2s, start, width, lam)
        if f < best_f
            best_f = f
            best_t = t
        end
    end
    # Golden-section refinement in the bracketing grid cell.
    h = (hi - start) / N
    lo_t = max(start, best_t - h)
    hi_t = min(hi, best_t + h)
    g = 0.6180339887498949
    x1 = hi_t - g * (hi_t - lo_t); x2 = lo_t + g * (hi_t - lo_t)
    f1 = _reg_cost(x1, a, r2s, start, width, lam); f2 = _reg_cost(x2, a, r2s, start, width, lam)
    for _ in 1:30
        if f1 < f2
            hi_t = x2; x2 = x1; f2 = f1
            x1 = hi_t - g * (hi_t - lo_t); f1 = _reg_cost(x1, a, r2s, start, width, lam)
        else
            lo_t = x1; x1 = x2; f1 = f2
            x2 = lo_t + g * (hi_t - lo_t); f2 = _reg_cost(x2, a, r2s, start, width, lam)
        end
    end
    t = f1 < f2 ? x1 : x2
    _reg_cost(t, a, r2s, start, width, lam) < best_f || (t = best_t)
    r = max(0.0, r0 * cos(t - a))
    return r * cis(copysign(t, φ0))
end

"""
Diagonal StefCal step with the phase prior: the per-antenna update is the
exact minimiser of den·|g - g_ls|² - λ log P(arg g) for each feed.
"""
function stefcal_step_diagonal_reg_kernel!(
    step_xx, step_yy, gain_xx, gain_yy,
    meas_xx, meas_xy, meas_yx, meas_yy,
    model_xx, model_xy, model_yx, model_yy,
    reg_start, reg_width, lam, Nant, Nfreq
)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        num_xx = ComplexF64(0); num_yy = ComplexF64(0)
        den_xx = ComplexF64(0); den_yy = ComplexF64(0)
        for i in 1:Nant
            gi_xx = gain_xx[i, β]; gi_yy = gain_yy[i, β]
            gm_xx = gi_xx * model_xx[i, j, β]; gm_xy = gi_xx * model_xy[i, j, β]
            gm_yx = gi_yy * model_yx[i, j, β]; gm_yy = gi_yy * model_yy[i, j, β]
            num_xx += conj(gm_xx) * meas_xx[i, j, β] + conj(gm_yx) * meas_yx[i, j, β]
            num_yy += conj(gm_xy) * meas_xy[i, j, β] + conj(gm_yy) * meas_yy[i, j, β]
            den_xx += conj(gm_xx) * gm_xx + conj(gm_yx) * gm_yx
            den_yy += conj(gm_xy) * gm_xy + conj(gm_yy) * gm_yy
        end
        new_xx = abs(den_xx) > eps(Float64) ? conj(num_xx / den_xx) : gain_xx[j, β]
        new_yy = abs(den_yy) > eps(Float64) ? conj(num_yy / den_yy) : gain_yy[j, β]
        st = reg_start[j, β]; wd = reg_width[j, β]
        new_xx = regularize_element(new_xx, real(den_xx), st, wd, lam)
        new_yy = regularize_element(new_yy, real(den_yy), st, wd, lam)
        step_xx[j, β] = new_xx - gain_xx[j, β]
        step_yy[j, β] = new_yy - gain_yy[j, β]
    end
    return nothing
end

"""
Full-Jones StefCal step with the phase prior on the diagonal Jones terms.
Writing X = Gⱼᴴ, each column of X has the quadratic cost (x-x₀)ᴴ D (x-x₀)
(D = Σ (GM)ᴴGM); minimising out the off-diagonal entry leaves the Schur
complement s|z - z₀|² for the diagonal entry, which is then regularized,
and the off-diagonal entry is set to its conditional optimum.
"""
function stefcal_step_full_reg_kernel!(
    step_xx, step_xy, step_yx, step_yy,
    gain_xx, gain_xy, gain_yx, gain_yy,
    meas_xx, meas_xy, meas_yx, meas_yy,
    model_xx, model_xy, model_yx, model_yy,
    reg_start, reg_width, lam, Nant, Nfreq
)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        num_xx = ComplexF64(0); num_xy = ComplexF64(0); num_yx = ComplexF64(0); num_yy = ComplexF64(0)
        den_xx = ComplexF64(0); den_xy = ComplexF64(0); den_yx = ComplexF64(0); den_yy = ComplexF64(0)
        for i in 1:Nant
            gi_xx = gain_xx[i, β]; gi_xy = gain_xy[i, β]; gi_yx = gain_yx[i, β]; gi_yy = gain_yy[i, β]
            m_xx = model_xx[i, j, β]; m_xy = model_xy[i, j, β]; m_yx = model_yx[i, j, β]; m_yy = model_yy[i, j, β]
            v_xx = meas_xx[i, j, β]; v_xy = meas_xy[i, j, β]; v_yx = meas_yx[i, j, β]; v_yy = meas_yy[i, j, β]
            gm_xx = gi_xx*m_xx + gi_xy*m_yx
            gm_xy = gi_xx*m_xy + gi_xy*m_yy
            gm_yx = gi_yx*m_xx + gi_yy*m_yx
            gm_yy = gi_yx*m_xy + gi_yy*m_yy
            num_xx += conj(gm_xx)*v_xx + conj(gm_yx)*v_yx
            num_xy += conj(gm_xx)*v_xy + conj(gm_yx)*v_yy
            num_yx += conj(gm_xy)*v_xx + conj(gm_yy)*v_yx
            num_yy += conj(gm_xy)*v_xy + conj(gm_yy)*v_yy
            den_xx += conj(gm_xx)*gm_xx + conj(gm_yx)*gm_yx
            den_xy += conj(gm_xx)*gm_xy + conj(gm_yx)*gm_yy
            den_yx += conj(gm_xy)*gm_xx + conj(gm_yy)*gm_yx
            den_yy += conj(gm_xy)*gm_xy + conj(gm_yy)*gm_yy
        end
        det_den = den_xx*den_yy - den_xy*den_yx
        if abs(det_den) > eps(Float64)
            inv_det = 1.0 / det_den
            # X = D⁻¹ N   (X = new Gⱼᴴ)
            x11 = ( den_yy*num_xx - den_xy*num_yx) * inv_det
            x12 = ( den_yy*num_xy - den_xy*num_yy) * inv_det
            x21 = (-den_yx*num_xx + den_xx*num_yx) * inv_det
            x22 = (-den_yx*num_xy + den_xx*num_yy) * inv_det
            st = reg_start[j, β]; wd = reg_width[j, β]
            d11 = real(den_xx); d22 = real(den_yy); d12sq = abs2(den_xy)
            # Column 1 = (conj G_xx, conj G_xy): regularize G_xx.
            if d22 > 0.0
                s1 = d11 - d12sq / d22
                z = regularize_element(conj(x11), s1, st, wd, lam)
                δ1 = conj(z) - x11
                x11 += δ1
                x21 += -den_yx * δ1 / d22
            end
            # Column 2 = (conj G_yx, conj G_yy): regularize G_yy.
            if d11 > 0.0
                s2 = d22 - d12sq / d11
                z = regularize_element(conj(x22), s2, st, wd, lam)
                δ2 = conj(z) - x22
                x22 += δ2
                x12 += -den_xy * δ2 / d11
            end
            step_xx[j, β] = conj(x11) - gain_xx[j, β]
            step_xy[j, β] = conj(x21) - gain_xy[j, β]
            step_yx[j, β] = conj(x12) - gain_yx[j, β]
            step_yy[j, β] = conj(x22) - gain_yy[j, β]
        else
            step_xx[j, β] = ComplexF64(0); step_xy[j, β] = ComplexF64(0)
            step_yx[j, β] = ComplexF64(0); step_yy[j, β] = ComplexF64(0)
        end
    end
    return nothing
end

#==============================================================================#
#                        Experimental StefCal driver                           #
#==============================================================================#

_gains(cal::GPUCalibration) = (cal.xx, cal.xy, cal.yx, cal.yy)
_as_cal(g, cal::GPUCalibration) = GPUCalibration(g[1], g[2], g[3], g[4], cal.flags, cal.is_diagonal)

"""Compute steps k = newgain(g) - g (optionally regularized) for gains g."""
function _stefcal_steps!(k, g, cal, meas_sq, model_sq, meta, reg)
    Na = meta.Nant
    Nf = Nfreq(cal)
    blocks, threads = kernel_config_1d(Na * Nf)
    if reg === nothing
        if cal.is_diagonal
            gpu_stefcal_step_diagonal!(k[1], k[4], _as_cal(g, cal), meas_sq, model_sq, meta)
        else
            gpu_stefcal_step_full!(k[1], k[2], k[3], k[4], _as_cal(g, cal), meas_sq, model_sq, meta)
        end
        return k
    end
    if cal.is_diagonal
        @cuda blocks=blocks threads=threads stefcal_step_diagonal_reg_kernel!(
            k[1], k[4], g[1], g[4],
            meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy,
            model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy,
            reg.start, reg.width, reg.lambda, Na, Nf)
    else
        @cuda blocks=blocks threads=threads stefcal_step_full_reg_kernel!(
            k[1], k[2], k[3], k[4], g[1], g[2], g[3], g[4],
            meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy,
            model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy,
            reg.start, reg.width, reg.lambda, Na, Nf)
    end
    CUDA.synchronize()
    k
end

_chan_sq(a) = vec(Array(sum(abs2.(a), dims=1)))

"""
    gpu_stefcal_exp!(cal, measured, model, meta; maxiter, tolerance, minuvw,
                     solver=:damped, reg=nothing) -> (converged, niters, frac_converged)

`solver=:damped` is TTCalX's production iteration (half steps, one global
convergence test); `solver=:rk4` reproduces CPU TTCal's `solve!`: RK4 of the
StefCal step, `maxiter` iterations per channel and a per-channel test
‖g_new - g‖/‖g‖ < tolerance, after which that channel is frozen.  `reg`
(a `PhaseRegularization`) applies the Tukey phase prior in every step.
"""
function gpu_stefcal_exp!(calibration::GPUCalibration, measured::GPUVisibilities,
                          model::GPUVisibilities, meta::GPUMetadata;
                          maxiter::Int=20, tolerance::Float64=1e-3, minuvw::Float64=0.0,
                          solver::Symbol=:damped, reg=nothing)
    _is_gpu(measured) || error("gpu_stefcal_exp! needs GPU arrays")
    Na = meta.Nant
    wideband = Nfreq(calibration) == 1 && Nfreq(measured) > 1
    meas_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(measured), gpu=true)
    model_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(model), gpu=true)
    gpu_makesquare!(meas_sq, measured, meta)
    gpu_makesquare!(model_sq, model, meta)
    if minuvw > 0
        flag_short_baselines!(meas_sq, meta, minuvw)
        flag_short_baselines!(model_sq, meta, minuvw)
    end
    Nf = Nfreq(calibration)
    zero_like() = ntuple(_ -> CUDA.zeros(ComplexF64, Na, Nf), 4)
    g = _gains(calibration)
    diag = calibration.is_diagonal
    comps = diag ? (1, 4) : (1, 2, 3, 4)

    if solver == :damped
        k = zero_like()
        converged = false
        niters = 0
        for iter in 1:maxiter
            niters = iter
            _stefcal_steps!(k, g, calibration, meas_sq, model_sq, meta, reg)
            norm_step = sqrt(sum(sum(abs2, k[c]) for c in comps))
            norm_gain = sqrt(sum(sum(abs2, g[c]) for c in comps))
            converged = norm_step < tolerance * norm_gain
            for c in comps
                g[c] .+= 0.5 .* k[c]
            end
            converged && break
        end
        return converged, niters, converged ? 1.0 : 0.0
    elseif solver == :rk4
        k1, k2, k3, k4, x, gnew = zero_like(), zero_like(), zero_like(), zero_like(),
                                  zero_like(), zero_like()
        active = trues(Nf)
        niters = 0
        for iter in 1:maxiter
            any(active) || break
            niters = iter
            _stefcal_steps!(k1, g, calibration, meas_sq, model_sq, meta, reg)
            for c in comps; x[c] .= g[c] .+ 0.5 .* k1[c]; end
            _stefcal_steps!(k2, x, calibration, meas_sq, model_sq, meta, reg)
            for c in comps; x[c] .= g[c] .+ 0.5 .* k2[c]; end
            _stefcal_steps!(k3, x, calibration, meas_sq, model_sq, meta, reg)
            for c in comps; x[c] .= g[c] .+ k3[c]; end
            _stefcal_steps!(k4, x, calibration, meas_sq, model_sq, meta, reg)
            for c in comps
                gnew[c] .= g[c] .+ (k1[c] .+ 2 .* k2[c] .+ 2 .* k3[c] .+ k4[c]) ./ 6
            end
            dn = zeros(Nf); gn = zeros(Nf)
            for c in comps
                dn .+= _chan_sq(gnew[c] .- g[c])
                gn .+= _chan_sq(g[c])
            end
            mask = CuArray(reshape(active, 1, Nf))
            for c in comps
                g[c] .= ifelse.(mask, gnew[c], g[c])
            end
            active .&= .!(sqrt.(dn) ./ sqrt.(gn) .< tolerance)
        end
        frac = 1 - count(active) / Nf
        return !any(active), niters, frac
    else
        error("unknown solver $solver (use :damped or :rk4)")
    end
end

"""Write a calibration's four Jones components as little-endian ComplexF64 (Nant, Nfreq, 4)."""
function save_calibration_bin(path::String, cal::GPUCalibration)
    h = cat(Array(cal.xx), Array(cal.xy), Array(cal.yx), Array(cal.yy); dims=3)
    open(path, "w") do io
        write(io, htol.(reinterpret(Float64, vec(h))))
    end
    size(h)
end

"""
    zero_short_baselines!(vis, meta, minuvw)

Zero a model's entries on baselines shorter than `minuvw` wavelengths.  CPU
TTCal's `peel!` flags those baselines in the dataset before peeling, so its
subtract!/add! skip them and they leave zest untouched; applying this to each
coherency reproduces that (TTCalX otherwise subtracts the gain-corrupted
model on short baselines whose gains were never constrained by the solve).
"""
function zero_short_baselines!(vis::GPUVisibilities, meta::GPUMetadata, minuvw::Float64)
    minuvw <= 0 && return vis
    pos = Array(meta.antenna_positions)
    bl = Array(meta.baselines)
    ch = Array(meta.channels)
    mask = trues(meta.Nbase, meta.Nfreq)
    for α in 1:meta.Nbase
        a1, a2 = bl[1, α], bl[2, α]
        len = sqrt((pos[1, a1] - pos[1, a2])^2 + (pos[2, a1] - pos[2, a2])^2 + (pos[3, a1] - pos[3, a2])^2)
        for β in 1:meta.Nfreq
            len < minuvw * SPEED_OF_LIGHT / ch[β] && (mask[α, β] = false)
        end
    end
    m = _is_gpu(vis) ? CuArray(mask) : mask
    vis.xx .*= m; vis.xy .*= m; vis.yx .*= m; vis.yy .*= m
    vis
end
