# Tukey phase-prior regularization of direction-dependent gains
# (R. Byrne, "Regularized Peeling", 2026; CaliCo `cost_ddcal`).

using LinearAlgebra

_mean(x) = sum(x) / length(x)
_median(x) = (v = sort(vec(x)); n = length(v); isodd(n) ? v[(n + 1) ÷ 2] : (v[n ÷ 2] + v[n ÷ 2 + 1]) / 2)

"""
Per-gain phase prior.  A source position error Δθ appears as an antenna
phase 2π sin(Δθ) ν ρ_j / c (ρ_j = planar distance from the array centre), so
the prior on each diagonal Jones term is

    P(φ) = 1                                      |φ| ≤ φ_start
         = ½ cos(π (|φ| - φ_start)/φ_width) + ½    φ_start < |φ| < φ_start + φ_width
         = 0                                      otherwise

with φ_start from `max_offset_deg` and φ_start + φ_width from
`max_offset_deg + taper_deg`.  Gains with φ_start ≥ π are left free.  The
solve minimises Σ|V - GMGᴴ|² - λ Σ log P(arg g).

The note's Eq. 9 and CaliCo's `tukey_taper` put 2π and the signed phase in
the cosine, which makes the taper non-monotonic and asymmetric; the intended
π / |φ| form is used here.
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

"""
    phase_regularization(meta, max_offset_deg, taper_deg; lambda=1.0, gpu=true, nfreq=meta.Nfreq)

Build the per-(antenna, channel) prior.  `lambda` weights the prior against
the unit-weight visibility cost.  `nfreq = 1` gives one prior per antenna at
the mean frequency (wideband solves).
"""
function phase_regularization(meta::GPUMetadata, max_offset_deg::Real, taper_deg::Real;
                              lambda::Real=1.0, gpu::Bool=true, nfreq::Integer=meta.Nfreq)
    nfreq in (1, meta.Nfreq) || throw(ArgumentError("nfreq must be 1 or meta.Nfreq"))
    dist = antenna_planar_distances(Array(meta.antenna_positions))
    channels = Array(meta.channels)
    nfreq == 1 && (channels = [_mean(channels)])
    k = 2π .* dist .* channels' ./ SPEED_OF_LIGHT
    start = k .* sin(deg2rad(max_offset_deg))
    stop = k .* sin(deg2rad(max_offset_deg + taper_deg))
    width = max.(stop .- start, 1e-12)
    start[start .>= π] .= Inf
    up = gpu && CUDA.functional() ? CuArray : identity
    PhaseRegularization(up(start), up(width), Float64(lambda), Float64(max_offset_deg), Float64(taper_deg))
end

"""Fraction of (antenna, channel) gains the prior can act on."""
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
    regularize_element(z0, s, start, width, lam) -> z

Minimise s|z - z0|² - λ log P(arg z) over complex z: exact in the modulus
(the best z at phase t is |z0| cos(t - arg z0) e^{it}), grid plus
golden-section search in the phase.  Returns z0 when the prior is inactive
(start = Inf, λ ≤ 0, s ≤ 0) or already satisfied.
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
