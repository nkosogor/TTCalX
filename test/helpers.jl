# Shared fixtures for the unit tests: a synthetic OVRO-LWA-like array.

using LinearAlgebra
using Random
using StaticArrays

const OVRO_ITRF = wgs84_to_itrf(-118.2817, 37.2398, 1183.0)
const TEST_MJD = 60664.125      # 2024-12-20 03:00 UTC

"""Local east, north, up unit vectors (ITRF) at the array centre."""
function enu_basis(center=OVRO_ITRF)
    up = normalize(Vector(center))
    east = normalize(cross([0.0, 0.0, 1.0], up))
    north = cross(up, east)
    east, north, up
end

"""J2000 (ra, dec) of the zenith at `mjd`."""
function zenith_radec(mjd=TEST_MJD)
    _, _, up = enu_basis()
    s = j2000_to_itrf_matrix(mjd)' * up
    atan(s[2], s[1]), asin(s[3])
end

"""J2000 (ra, dec) of the direction at (azimuth, elevation) in degrees at `mjd`."""
function azel_radec(az_deg, el_deg; mjd=TEST_MJD)
    east, north, up = enu_basis()
    az, el = deg2rad(az_deg), deg2rad(el_deg)
    d = cos(el) * (sin(az) * east + cos(az) * north) + sin(el) * up
    s = j2000_to_itrf_matrix(mjd)' * d
    atan(s[2], s[1]), asin(s[3])
end

"""
    synthetic_meta(; Nant, channels, radius, seed, gpu, mjd) -> meta

Random antennas within ~`radius` m of the OVRO-LWA site, all cross baselines
(a1 < a2), phase centre at the zenith, and UVW in the OVRO-LWA MS convention
(r₁ - r₂, unprecessed: w = -ŝ₀·b).
"""
function synthetic_meta(; Nant=24, channels=[40e6, 55e6, 70e6], radius=800.0, seed=1,
                        gpu=false, mjd=TEST_MJD)
    rng = MersenneTwister(seed)
    east, north, up = enu_basis()
    pos = reduce(hcat, [Vector(OVRO_ITRF) .+ radius .* (randn(rng) .* east .+ randn(rng) .* north) ./ 2 .+
                        0.5 .* randn(rng) .* up for _ in 1:Nant])
    pairs = [(i, j) for i in 1:Nant for j in i+1:Nant]
    bl = Int32.(reduce(hcat, [[i, j] for (i, j) in pairs]))
    ra, dec = zenith_radec(mjd)
    s0 = j2000_to_itrf_matrix(mjd) * SVector(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec))
    uvw = zeros(3, length(pairs))
    for (α, (i, j)) in enumerate(pairs)
        b = pos[:, j] .- pos[:, i]
        uvw[1, α] = -dot(b, east); uvw[2, α] = -dot(b, north); uvw[3, α] = -dot(b, s0)
    end
    create_gpu_metadata(pos, bl, Float64.(channels), [0.0, 0.0, 1.0], uvw;
                        phase_center_ra=ra, phase_center_dec=dec, gpu=gpu)
end

"""Host visibilities with standard-normal complex entries (and optional random flags)."""
function random_vis(Nb, Nf; seed=2, flag_fraction=0.0)
    rng = MersenneTwister(seed)
    r() = randn(rng, ComplexF64, Nb, Nf)
    GPUVisibilities(r(), r(), r(), r(), rand(rng, Nb, Nf) .< flag_fraction)
end

"""Full-Jones host gains I + ε·randn."""
function random_gains(Na, Nf; seed=3, ε=0.1)
    rng = MersenneTwister(seed)
    r() = ε .* randn(rng, ComplexF64, Na, Nf)
    GPUCalibration(1 .+ r(), r(), r(), 1 .+ r(), falses(Na, Nf), false)
end

host_copy(v::GPUVisibilities) = GPUVisibilities(Array(v.xx), Array(v.xy), Array(v.yx), Array(v.yy), Array(v.flags))
host_copy(c::GPUCalibration) = GPUCalibration(Array(c.xx), Array(c.xy), Array(c.yx), Array(c.yy), Array(c.flags), c.is_diagonal)

"""Max |a - b| / max |b| over the four polarizations."""
relmax(a::GPUVisibilities, b::GPUVisibilities) =
    maximum(maximum(abs, Array(getfield(a, f)) .- Array(getfield(b, f))) for f in (:xx, :xy, :yx, :yy)) /
    maximum(maximum(abs, Array(getfield(b, f))) for f in (:xx, :xy, :yx, :yy))

"""‖V - J₁ M J₂ᴴ‖ / ‖V‖ over unflagged visibilities."""
function fit_residual(measured, model, cal, meta)
    m = deepcopy_gpu(model)
    gpu_corrupt!(m, cal, meta)
    ok = .!Array(measured.flags)
    num = sum(sum(abs2, (Array(getfield(measured, f)) .- Array(getfield(m, f)))[ok]) for f in (:xx, :xy, :yx, :yy))
    den = sum(sum(abs2, Array(getfield(measured, f))[ok]) for f in (:xx, :xy, :yx, :yy))
    sqrt(num / den)
end
