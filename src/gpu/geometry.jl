# Apparent (precessed) array geometry, as CPU TTCal uses it.
#
# TTCalX's far-field kernels take UVW from the MS and source l, m, n in J2000
# about PHASE_DIR.  OVRO-LWA MSes store UVW = r₁ - r₂ without precession, so
# the kernel model is the mirror image of the sky and ~0.35° off; free
# full-Jones gains hide this in plain peeling.  `geometric_uvw!` replaces the
# UVW with ITRF baselines rotated into J2000, which gives CPU TTCal's
# apparent geometry k·b·(ŝ - ŝ₀).

using LinearAlgebra
using StaticArrays

const SPEED_OF_LIGHT = 299792458.0

"""Unit vector of J2000 (ra, dec) in radians."""
_unit(ra, dec) = SVector(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec))

_rx(a) = @SMatrix [1.0 0.0 0.0; 0.0 cos(a) sin(a); 0.0 -sin(a) cos(a)]
_ry(a) = @SMatrix [cos(a) 0.0 -sin(a); 0.0 1.0 0.0; sin(a) 0.0 cos(a)]
_rz(a) = @SMatrix [cos(a) sin(a) 0.0; -sin(a) cos(a) 0.0; 0.0 0.0 1.0]

"""
    j2000_to_itrf_matrix(mjd_utc) -> R   (x_itrf = R * x_j2000)

IAU 1976 precession, the leading IAU 1980 nutation terms and GAST (UT1 ≈ UTC).
Polar motion, frame bias and aberration are ignored (≲ 20″), far below the
OVRO-LWA resolution; precession since J2000 (~0.35° in 2024) is not.
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

"""
    uvw_geometry(meta) -> (s0, sign, rms)

Phase-centre unit vector ŝ₀ (ITRF) and the UVW sign convention of `meta.uvw`,
from a least-squares fit of the `w` column to the ITRF baselines:
w_α = sign · ŝ₀·(r₂ - r₁).  Snapshots are phased near zenith, so ŝ₀ is the
solution pointing up.  OVRO-LWA MSes have sign = -1 (UVW = r₁ - r₂).  `rms`
is the fit residual (m).
"""
function uvw_geometry(meta::GPUMetadata)
    pos = Array(meta.antenna_positions)
    bl = Array(meta.baselines)
    w = Array(meta.uvw)[3, :]
    sel = findall(bl[1, :] .!= bl[2, :])
    B = permutedims(pos[:, bl[2, sel]] .- pos[:, bl[1, sel]])
    v = B \ w[sel]
    up = vec(sum(pos, dims=2))
    up = up / norm(up)
    sign = dot(v, up) >= 0 ? 1 : -1
    s0 = SVector{3}(sign .* v ./ norm(v))
    rms = sqrt(sum(abs2, B * v .- w[sel]) / length(sel))
    s0, sign, rms
end

"""
    geometric_uvw!(meta, mjd_utc) -> (rms_w, phase_centre_offset_deg)

Replace `meta.uvw` with the ITRF baselines r₂ - r₁ rotated into J2000
(`j2000_to_itrf_matrix`) and projected on the (east, north, PHASE_DIR)
basis.  With the J2000 l, m, n of TTCalX's kernels the far-field phase then
follows the apparent geometry, as in CPU TTCal.  Returns two sanity checks:
the rms difference from the original `w` (in the original sign convention)
and the angle between the rotated PHASE_DIR and the ŝ₀ fitted from it.
"""
function geometric_uvw!(meta::GPUMetadata, mjd_utc::Float64)
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
        uvw[:, α] = E * b
    end
    rms_w = sqrt(sum(abs2, uvw[3, :] .- msign .* w_ms) / size(bl, 2))
    copyto!(meta.uvw, uvw)
    rms_w, offset
end
