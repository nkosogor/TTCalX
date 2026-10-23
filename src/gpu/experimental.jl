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
    hxx = zeros(ComplexF64, Nb, Nf); hxy = zeros(ComplexF64, Nb, Nf)
    hyx = zeros(ComplexF64, Nb, Nf); hyy = zeros(ComplexF64, Nb, Nf)
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
    if !_is_gpu(cal)
        reg === nothing || error("the phase prior is only implemented for GPU arrays")
        return _host_stefcal_steps!(k, g, cal, meas_sq, model_sq, meta)
    end
    Na = meta.Nant
    Nf = Nfreq(cal)
    blocks, threads = kernel_config_1d(Na * Nf)
    if reg === nothing
        if cal.is_diagonal
            gpu_stefcal_step_diagonal!(k[1], k[4], _as_cal(g, cal), meas_sq, model_sq, meta)
        else
            @cuda blocks=blocks threads=threads stefcal_step_full_herm_kernel!(
                k[1], k[2], k[3], k[4], g[1], g[2], g[3], g[4],
                meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy,
                model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy, Na, Nf)
            CUDA.synchronize()
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
Host arrays run the same steps on host threads (without `reg`).
"""
function gpu_stefcal_exp!(calibration::GPUCalibration, measured::GPUVisibilities,
                          model::GPUVisibilities, meta::GPUMetadata;
                          maxiter::Int=20, tolerance::Float64=1e-3, minuvw::Float64=0.0,
                          solver::Symbol=:damped, reg=nothing)
    use_gpu = _is_gpu(measured)
    Na = meta.Nant
    wideband = Nfreq(calibration) == 1 && Nfreq(measured) > 1
    !use_gpu && wideband && error("host gpu_stefcal_exp! is per-channel only")
    meas_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(measured), gpu=use_gpu)
    model_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(model), gpu=use_gpu)
    makesquare! = use_gpu ? gpu_makesquare! : host_makesquare!
    makesquare!(meas_sq, measured, meta)
    makesquare!(model_sq, model, meta)
    if minuvw > 0
        flag_short! = use_gpu ? flag_short_baselines_gpu! : flag_short_baselines!
        flag_short!(meas_sq, meta, minuvw)
        flag_short!(model_sq, meta, minuvw)
    end
    Nf = Nfreq(calibration)
    zero_like() = ntuple(_ -> use_gpu ? CUDA.zeros(ComplexF64, Na, Nf) : zeros(ComplexF64, Na, Nf), 4)
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
            mask = use_gpu ? CuArray(reshape(active, 1, Nf)) : reshape(active, 1, Nf)
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

#==============================================================================#
#              CPU-TTCal-exact far-field model (host computation)               #
#==============================================================================#

_components(s::GPUMultiSource) = s.components
_components(s::GPUSource) = GPUSource[s]

"""Gaussian axes in ITRF and widths, as CPU TTCal `additional_precomputation`."""
function _gaussian_axes(c::GPUGaussianSource, R)
    s = _unit(c.ra, c.dec)
    z = SVector(0.0, 0.0, 1.0)
    north = z - dot(z, s) * s
    north = north / norm(north)
    east = cross(north, s)
    θ = c.position_angle
    major = R * (cos(θ) * north + sin(θ) * east)
    minor = R * (-sin(θ) * north + cos(θ) * east)
    wmaj = π^2 * sin(c.major_fwhm)^2 / (4log(2))
    wmin = π^2 * sin(c.minor_fwhm)^2 / (4log(2))
    major, minor, wmaj, wmin
end

"""
    genvis_exact!(vis, meta, source, mjd_utc)

Add a sky source's model the way CPU TTCal does (`genvis!` with a constant
beam): per component, skipped below the horizon; fringe exp(-ik (r₂-r₁)·(ŝ-ŝ₀))
with apparent ITRF directions (`j2000_to_itrf_matrix`); Gaussian coherency
from the 3-D baseline projected on the component's major/minor axes built
from its local north/east (TTCalX's kernels use the (u,v) projection, a PA
measured from east and a different rotation, and no foreshortening); and
Stokes → linear with xy = U - iV.  Computed on the host, then uploaded.

Reference implementation (single-threaded; `c` is abstractly typed, so the
inner loop is dynamically dispatched).  `genvis_exact!` computes the same
model with a GPU kernel or host threads.
"""
function genvis_exact_reference!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUSource, mjd_utc::Float64)
    R = j2000_to_itrf_matrix(mjd_utc)
    pos = Array(meta.antenna_positions)
    bl = Array(meta.baselines)
    ch = Array(meta.channels)
    Nb, Nf = meta.Nbase, meta.Nfreq
    s0 = R * _unit(meta.phase_center_ra, meta.phase_center_dec)
    up = vec(sum(pos, dims=2)); up = up / norm(up)
    hxx = zeros(ComplexF64, Nb, Nf); hxy = zeros(ComplexF64, Nb, Nf)
    hyx = zeros(ComplexF64, Nb, Nf); hyy = zeros(ComplexF64, Nb, Nf)
    for c in _components(source)
        s = R * _unit(c.ra, c.dec)
        dot(s, up) > 0 || continue
        gauss = c isa GPUGaussianSource ? _gaussian_axes(c, R) : nothing
        delay = [dot(SVector(pos[1, i], pos[2, i], pos[3, i]), s - s0) for i in axes(pos, 2)]
        for β in 1:Nf
            I, Q, U, V = c.spectrum(ch[β])
            fxx, fxy, fyx, fyy = complex(I + Q), complex(U, -V), complex(U, V), complex(I - Q)
            k = 2π * ch[β] / SPEED_OF_LIGHT
            λ = SPEED_OF_LIGHT / ch[β]
            @inbounds for α in 1:Nb
                a1, a2 = bl[1, α], bl[2, α]
                v = cis(-k * (delay[a2] - delay[a1]))
                if gauss !== nothing
                    major, minor, wmaj, wmin = gauss
                    b = SVector(pos[1, a2] - pos[1, a1], pos[2, a2] - pos[2, a1], pos[3, a2] - pos[3, a1])
                    v *= exp(-wmaj * (dot(b, major) / λ)^2 - wmin * (dot(b, minor) / λ)^2)
                end
                hxx[α, β] += fxx * v; hxy[α, β] += fxy * v
                hyx[α, β] += fyx * v; hyy[α, β] += fyy * v
            end
        end
    end
    if _is_gpu(vis)
        vis.xx .+= CuArray(hxx); vis.xy .+= CuArray(hxy); vis.yx .+= CuArray(hyx); vis.yy .+= CuArray(hyy)
    else
        vis.xx .+= hxx; vis.xy .+= hxy; vis.yx .+= hyx; vis.yy .+= hyy
    end
    vis
end

"""
Per-component tables for `genvis_exact!`: geometric delay (m) per antenna,
linear-feed fluxes per channel, and Gaussian axes/widths (zero for points).
Components below the horizon are dropped.
"""
function _exact_tables(meta::GPUMetadata, source::GPUSource, mjd_utc::Float64)
    R = j2000_to_itrf_matrix(mjd_utc)
    pos = Array(meta.antenna_positions)
    ch = Array(meta.channels)
    s0 = R * _unit(meta.phase_center_ra, meta.phase_center_dec)
    up = vec(sum(pos, dims=2)); up = up / norm(up)
    comps = [c for c in _components(source) if dot(R * _unit(c.ra, c.dec), up) > 0]
    Na, Nf, Nc = size(pos, 2), length(ch), length(comps)
    delay = zeros(Na, Nc)
    flux = zeros(ComplexF64, 4, Nf, Nc)
    shape = zeros(9, Nc)   # major (3), minor (3), wmaj, wmin, is_gaussian
    for (n, c) in enumerate(comps)
        s = R * _unit(c.ra, c.dec)
        for i in 1:Na
            delay[i, n] = dot(SVector(pos[1, i], pos[2, i], pos[3, i]), s - s0)
        end
        for β in 1:Nf
            I, Q, U, V = c.spectrum(ch[β])
            flux[1, β, n] = complex(I + Q); flux[2, β, n] = complex(U, -V)
            flux[3, β, n] = complex(U, V); flux[4, β, n] = complex(I - Q)
        end
        if c isa GPUGaussianSource
            major, minor, wmaj, wmin = _gaussian_axes(c, R)
            shape[1:3, n] .= major; shape[4:6, n] .= minor
            shape[7, n] = wmaj; shape[8, n] = wmin; shape[9, n] = 1
        end
    end
    delay, flux, shape
end

"""Model at one (baseline, channel), summed over components (same arithmetic as the reference)."""
@inline function _exact_vis(α, β, delay, flux, shape, positions, baselines, channels, Nc)
    a1 = baselines[1, α]; a2 = baselines[2, α]
    ν = channels[β]
    k = 2π * ν / SPEED_OF_LIGHT
    λ = SPEED_OF_LIGHT / ν
    bx = positions[1, a2] - positions[1, a1]
    by = positions[2, a2] - positions[2, a1]
    bz = positions[3, a2] - positions[3, a1]
    axx = zero(ComplexF64); axy = zero(ComplexF64); ayx = zero(ComplexF64); ayy = zero(ComplexF64)
    for n in 1:Nc
        v = cis(-k * (delay[a2, n] - delay[a1, n]))
        if shape[9, n] != 0
            pmaj = (bx * shape[1, n] + by * shape[2, n] + bz * shape[3, n]) / λ
            pmin = (bx * shape[4, n] + by * shape[5, n] + bz * shape[6, n]) / λ
            v *= exp(-shape[7, n] * pmaj^2 - shape[8, n] * pmin^2)
        end
        axx += flux[1, β, n] * v; axy += flux[2, β, n] * v
        ayx += flux[3, β, n] * v; ayy += flux[4, β, n] * v
    end
    axx, axy, ayx, ayy
end

function genvis_exact_kernel!(xx, xy, yx, yy, delay, flux, shape, positions, baselines, channels,
                              Nbase, Nfreq, Nc)
    idx = thread_index_1d()
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        vxx, vxy, vyx, vyy = _exact_vis(α, β, delay, flux, shape, positions, baselines, channels, Nc)
        xx[α, β] += vxx; xy[α, β] += vxy; yx[α, β] += vyx; yy[α, β] += vyy
    end
    return nothing
end

"""
    genvis_exact!(vis, meta, source, mjd_utc)

CPU-TTCal-exact far-field model (see `genvis_exact_reference!` for the
definition): the per-component tables are built on the host (Na delays,
Nf fluxes), then every (baseline, channel) is evaluated by a GPU kernel,
or by host threads for host arrays.
"""
function genvis_exact!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUSource, mjd_utc::Float64)
    delay, flux, shape = _exact_tables(meta, source, mjd_utc)
    Nc = size(delay, 2)
    Nc == 0 && return vis
    Nb, Nf = meta.Nbase, meta.Nfreq
    if _is_gpu(vis)
        blocks, threads = kernel_config_1d(Nb * Nf)
        @cuda blocks=blocks threads=threads genvis_exact_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy, CuArray(delay), CuArray(flux), CuArray(shape),
            meta.antenna_positions, meta.baselines, meta.channels, Nb, Nf, Nc)
        CUDA.synchronize()
    else
        pos, bl, ch = meta.antenna_positions, meta.baselines, meta.channels
        Threads.@threads for β in 1:Nf
            @inbounds for α in 1:Nb
                vxx, vxy, vyx, vyy = _exact_vis(α, β, delay, flux, shape, pos, bl, ch, Nc)
                vis.xx[α, β] += vxx; vis.xy[α, β] += vxy
                vis.yx[α, β] += vyx; vis.yy[α, β] += vyy
            end
        end
    end
    vis
end

#==============================================================================#
#              Host (CPU) backend: multithreaded, same arithmetic               #
#==============================================================================#
# TTCalX's peel operations launch CUDA kernels unconditionally.  These methods
# run the same per-element arithmetic on host arrays, one channel per thread
# (start Julia with -t N), so the daemon can zest with ZEST_DEVICE=cpu.

const HostVisibilities = GPUVisibilities{<:Array}

_channel(a::AbstractMatrix, β) = view(a, :, β:β)
_channel(a::AbstractArray{<:Any,3}, β) = view(a, :, :, β:β)
_channel(v::GPUVisibilities, β) = GPUVisibilities(_channel(v.xx, β), _channel(v.xy, β),
                                                  _channel(v.yx, β), _channel(v.yy, β), _channel(v.flags, β))
_channel(c::GPUCalibration, β) = GPUCalibration(_channel(c.xx, β), _channel(c.xy, β), _channel(c.yx, β),
                                                _channel(c.yy, β), _channel(c.flags, β), c.is_diagonal)
_channel(s::GPUSquareVisibilities, β) = GPUSquareVisibilities(_channel(s.xx, β), _channel(s.xy, β),
                                                              _channel(s.yx, β), _channel(s.yy, β))

function gpu_corrupt!(vis::HostVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nfreq(cal) == 1 && return cpu_corrupt!(vis, cal, meta)
    Threads.@threads for β in 1:Nfreq(vis)
        cpu_corrupt!(_channel(vis, β), _channel(cal, β), meta)
    end
    vis
end

gpu_subsrc!(data::HostVisibilities, model::GPUVisibilities) = cpu_subsrc!(data, model)
gpu_putsrc!(data::HostVisibilities, model::GPUVisibilities) = cpu_putsrc!(data, model)

"""As the `makesquare_kernel!` (not `cpu_makesquare!`, which keeps autos and accumulates)."""
function host_makesquare!(sq::GPUSquareVisibilities, vis::GPUVisibilities, meta::GPUMetadata)
    bl = meta.baselines
    Threads.@threads for β in 1:Nfreq(vis)
        for a in (sq.xx, sq.xy, sq.yx, sq.yy)
            fill!(view(a, :, :, β), zero(ComplexF64))
        end
        @inbounds for α in 1:meta.Nbase
            vis.flags[α, β] && continue
            a1, a2 = bl[1, α], bl[2, α]
            a1 == a2 && continue
            vxx, vxy, vyx, vyy = vis.xx[α, β], vis.xy[α, β], vis.yx[α, β], vis.yy[α, β]
            sq.xx[a1, a2, β] = vxx; sq.xy[a1, a2, β] = vxy
            sq.yx[a1, a2, β] = vyx; sq.yy[a1, a2, β] = vyy
            sq.xx[a2, a1, β] = conj(vxx); sq.xy[a2, a1, β] = conj(vyx)
            sq.yx[a2, a1, β] = conj(vxy); sq.yy[a2, a1, β] = conj(vyy)
        end
    end
    sq
end

"""StefCal steps on the host: `cpu_stefcal_step_*!` (the kernels' arithmetic) per channel thread."""
function _host_stefcal_steps!(k, g, cal, meas_sq, model_sq, meta)
    c = _as_cal(g, cal)
    Threads.@threads for β in 1:Nfreq(cal)
        kc = map(x -> _channel(x, β), k)
        if cal.is_diagonal
            cpu_stefcal_step_diagonal!(kc[1], kc[4], _channel(c, β), _channel(meas_sq, β),
                                       _channel(model_sq, β), meta)
        else
            cpu_stefcal_step_full!(kc[1], kc[2], kc[3], kc[4], _channel(c, β), _channel(meas_sq, β),
                                   _channel(model_sq, β), meta)
        end
    end
    k
end

"""
    flag_short_baselines_gpu!(sq, meta, minuvw)

`flag_short_baselines!` without the host round trip (it copies all four
Nant×Nant×Nfreq matrices to the host and back, ~40% of a zest solve):
zero every entry whose baseline is shorter than `minuvw` wavelengths.
"""
function flag_short_baselines_gpu!(sq::GPUSquareVisibilities, meta::GPUMetadata, minuvw::Float64)
    minuvw <= 0 && return sq
    pos = meta.antenna_positions
    L = sqrt.((pos[1, :] .- pos[1, :]').^2 .+ (pos[2, :] .- pos[2, :]').^2 .+ (pos[3, :] .- pos[3, :]').^2)
    Nf = size(sq.xx, 3)
    ch = meta.channels[1:min(Nf, length(meta.channels))]
    keep = reshape(L, size(L)..., 1) .>= reshape(minuvw .* (SPEED_OF_LIGHT ./ ch), 1, 1, :)
    # the host loop skips ant1 == ant2; those entries are zero after makesquare
    for a in (sq.xx, sq.xy, sq.yx, sq.yy)
        a .= ifelse.(keep, a, zero(ComplexF64))
    end
    sq
end

"""
`stefcal_step_full_kernel!` reading the square matrices by columns.  The
square visibilities are Hermitian (makesquare fills [a2,a1] with the
conjugate transpose of [a1,a2]), so M[i,j] = (M[j,i])ᴴ: xx = conj(xx[j,i]),
xy = conj(yx[j,i]), yx = conj(xy[j,i]), yy = conj(yy[j,i]).  Thread j then
reads [j,i], adjacent threads read adjacent addresses (coalesced) instead of
addresses Nant apart; the arithmetic, and so the result, is unchanged.
"""
function stefcal_step_full_herm_kernel!(
    step_xx, step_xy, step_yx, step_yy,
    gain_xx, gain_xy, gain_yx, gain_yy,
    meas_xx, meas_xy, meas_yx, meas_yy,
    model_xx, model_xy, model_yx, model_yy,
    Nant, Nfreq
)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        num_xx = ComplexF64(0); num_xy = ComplexF64(0); num_yx = ComplexF64(0); num_yy = ComplexF64(0)
        den_xx = ComplexF64(0); den_xy = ComplexF64(0); den_yx = ComplexF64(0); den_yy = ComplexF64(0)
        for i in 1:Nant
            gi_xx = gain_xx[i, β]; gi_xy = gain_xy[i, β]; gi_yx = gain_yx[i, β]; gi_yy = gain_yy[i, β]
            m_xx = conj(model_xx[j, i, β]); m_xy = conj(model_yx[j, i, β])
            m_yx = conj(model_xy[j, i, β]); m_yy = conj(model_yy[j, i, β])
            v_xx = conj(meas_xx[j, i, β]); v_xy = conj(meas_yx[j, i, β])
            v_yx = conj(meas_xy[j, i, β]); v_yy = conj(meas_yy[j, i, β])
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
            inv_xx = den_yy * inv_det; inv_xy = -den_xy * inv_det
            inv_yx = -den_yx * inv_det; inv_yy = den_xx * inv_det
            sol_xx = inv_xx*num_xx + inv_xy*num_yx
            sol_xy = inv_xx*num_xy + inv_xy*num_yy
            sol_yx = inv_yx*num_xx + inv_yy*num_yx
            sol_yy = inv_yx*num_xy + inv_yy*num_yy
            step_xx[j, β] = conj(sol_xx) - gain_xx[j, β]
            step_xy[j, β] = conj(sol_yx) - gain_xy[j, β]
            step_yx[j, β] = conj(sol_xy) - gain_yx[j, β]
            step_yy[j, β] = conj(sol_yy) - gain_yy[j, β]
        else
            step_xx[j, β] = ComplexF64(0); step_xy[j, β] = ComplexF64(0)
            step_yx[j, β] = ComplexF64(0); step_yy[j, β] = ComplexF64(0)
        end
    end
    return nothing
end

#==============================================================================#
#                       Faster MS write-back (numpy)                            #
#==============================================================================#

const _ms_writer = PyNULL()

const _MS_WRITER_PY = """
import numpy as np
from casacore.tables import table

def write(ms_path, column, index, xx, xy, yx, yy, flags):
    t = table(ms_path, readonly=False, ack=False)
    try:
        a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2")
        b = index[np.minimum(a1, a2), np.maximum(a1, a2)] - 1
        b[a1 == a2] = -1
        rows = np.nonzero(b >= 0)[0]
        b = b[rows]
        if len(np.unique(b)) != len(b):
            return False          # several rows per baseline: use write_gpu_to_ms!
        data = t.getcol(column)
        if data.ndim != 3 or data.shape[2] != 4:
            return False
        flag = t.getcol("FLAG")
        for p, v in enumerate((xx, xy, yx, yy)):
            data[rows, :, p] = v[b, :]
            flag[rows, :, p] = flags[b, :]
        t.putcol(column, data)
        t.putcol("FLAG", flag)
        t.flush()
        return True
    finally:
        t.close()
"""

"""
    write_ms_fast!(ms_path, vis, baseline_dict, Nrows; column="DATA")

Same result as `write_gpu_to_ms!` (cross-correlation DATA replaced, FLAG
set to the per-baseline flag in all four polarizations, autos untouched),
but the row scatter runs in numpy on the arrays casacore returns, without
converting the column to a Julia array and back.  Falls back to
`write_gpu_to_ms!` when a baseline has several rows or the MS is not 4-pol.
"""
function write_ms_fast!(ms_path::String, vis::GPUVisibilities, baseline_dict::Dict, Nrows::Int;
                        column::String="DATA")
    if _ms_writer == PyNULL()
        mod = PyCall.pyimport("types").ModuleType("ttcalx_ms_writer")
        PyCall.pybuiltin("exec")(_MS_WRITER_PY, mod."__dict__")
        copy!(_ms_writer, mod.write)
    end
    Na = maximum(maximum(k) for k in keys(baseline_dict))
    index = zeros(Int, Na, Na)            # 0-based antennas in the MS → 1-based baseline
    for ((a1, a2), α) in baseline_dict
        index[a1, a2] = α
    end
    ok = _ms_writer(ms_path, column, PyCall.PyReverseDims(permutedims(index)),
                    PyCall.PyReverseDims(permutedims(Array(vis.xx))), PyCall.PyReverseDims(permutedims(Array(vis.xy))),
                    PyCall.PyReverseDims(permutedims(Array(vis.yx))), PyCall.PyReverseDims(permutedims(Array(vis.yy))),
                    PyCall.PyReverseDims(permutedims(Array(vis.flags))))
    ok || write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column=column)
    nothing
end
