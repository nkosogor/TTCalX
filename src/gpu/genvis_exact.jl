# Far-field sky model computed exactly as CPU TTCal's `genvis!` (constant beam).
#
# Differences from TTCalX's genvis kernels, which this model removes:
#   - apparent ITRF directions (precession, nutation, GAST) instead of J2000
#     l, m about PHASE_DIR with the MS UVW;
#   - each component of a multi-component source is skipped below the horizon;
#   - Gaussian axes from the component's local north/east, with the 3-D
#     baseline projected on them (the kernels project (u, v), measure the PA
#     from east, rotate by a different angle and have no foreshortening);
#   - Stokes → linear with xy = U - iV.

using LinearAlgebra
using StaticArrays

_components(s::GPUMultiSource) = s.components
_components(s::GPUSource) = GPUSource[s]

"""Gaussian major/minor axes (ITRF unit vectors) and widths, as CPU TTCal's `additional_precomputation`."""
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
    exact_model_tables(meta, source, mjd_utc) -> (delay, flux, shape)

Per-component inputs of the exact model, for the components above the
horizon (geocentric: ŝ·r̂_centre > 0, within ~0.2° of CPU TTCal's geodetic
test): geometric delay (m) per antenna `delay[i, c] = r_i·(ŝ_c - ŝ₀)`,
linear-feed fluxes per channel `flux[1:4, β, c]` (xx, xy, yx, yy) and
`shape[1:9, c]` = Gaussian major axis (3), minor axis (3), widths w_maj,
w_min and a Gaussian flag (all zero for point components).
"""
function exact_model_tables(meta::GPUMetadata, source::GPUSource, mjd_utc::Float64)
    R = j2000_to_itrf_matrix(mjd_utc)
    pos = Array(meta.antenna_positions)
    ch = Array(meta.channels)
    s0 = R * _unit(meta.phase_center_ra, meta.phase_center_dec)
    up = vec(sum(pos, dims=2))
    up = up / norm(up)
    comps = [c for c in _components(source) if dot(R * _unit(c.ra, c.dec), up) > 0]
    Na, Nf, Nc = size(pos, 2), length(ch), length(comps)
    delay = zeros(Na, Nc)
    flux = zeros(ComplexF64, 4, Nf, Nc)
    shape = zeros(9, Nc)
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
            shape[1:3, n] .= major
            shape[4:6, n] .= minor
            shape[7, n] = wmaj; shape[8, n] = wmin; shape[9, n] = 1
        end
    end
    delay, flux, shape
end

"""
Model at one (baseline α, channel β), summed over components:

    V = Σ_c S_c exp(-ik(τ₂ - τ₁)) G_c,   G_c = exp(-w_maj (b·ê_maj/λ)² - w_min (b·ê_min/λ)²)

with b = r₂ - r₁ and k = 2πν/c.  Shared by the GPU kernel and the host loop.
"""
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

genvis_exact!(::GPUVisibilities, ::GPUMetadata, ::GPURFISource, ::Float64) =
    throw(ArgumentError("genvis_exact! is for sky sources; use gpu_genvis! for RFI sources"))

"""
    genvis_exact!(vis, meta, source, mjd_utc)

Add the model of a sky source computed as CPU TTCal does (see the file
header).  `mjd_utc` is the observation time (`ms_mjd`).  The per-component
tables are built on the host; every (baseline, channel) is then evaluated by
a GPU kernel, or by host threads for host arrays.  `meta.uvw` is not used,
so the result does not depend on the MS's UVW convention.
"""
function genvis_exact!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUSource, mjd_utc::Float64)
    delay, flux, shape = exact_model_tables(meta, source, mjd_utc)
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
