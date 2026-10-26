# Near-field RFI sources: a port of CPU TTCal's `RFISource` (the ttcal_dev
# build the main pipeline uses for RFI peeling).

using LinearAlgebra
using StaticArrays

"""
Per-channel RFI spectrum.  Evaluation reproduces ttcal_dev's `RFISpectrum`:
`searchsortedfirst` on the channel list, clamping outside it and linear
interpolation between the bracketing channels.  The list is assumed sorted;
an unsorted model entry is reproduced, not repaired.
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

"""Terrestrial (near-field) RFI emitter at a fixed ITRF position (m)."""
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

"""WGS84 geodetic (deg, deg, m above the ellipsoid) to ITRF XYZ (m)."""
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

"""
Build a `GPURFISource` from a TTCal JSON entry (`sys`, `el`, `long`, `lat`,
`rfi-frequencies`, `rfi-I` and optionally `rfi-Q/U/V`).  With `sys = "WGS84"`
the position is geodetic (height `el` in m, `long`/`lat` in degrees); with
`"ITRF"` the three fields are the Cartesian X, Y, Z.
"""
function construct_gpu_rfi_source(c::Dict)
    name = get(c, "name", "")
    sys = get(c, "sys", "WGS84")
    el, lon, lat = Float64(c["el"]), Float64(c["long"]), Float64(c["lat"])
    position = sys == "WGS84" ? wgs84_to_itrf(lon, lat, el) :
               sys == "ITRF"  ? SVector(el, lon, lat) :
               error("RFI source '$name': unknown coordinate system $sys")
    channels = Float64.(c["rfi-frequencies"])
    N = length(channels)
    for k in ("rfi-I", "rfi-Q", "rfi-U", "rfi-V")
        haskey(c, k) && length(c[k]) != N &&
            throw(ArgumentError("RFI source '$name': $k has $(length(c[k])) values for $N frequencies"))
    end
    issorted(channels) || log_warning("RFI source '$name': rfi-frequencies are not sorted; " *
                                      "spectrum lookup reproduces CPU TTCal's behaviour")
    spectrum = GPURFISpectrum(channels, Float64.(c["rfi-I"]),
                              Float64.(get(c, "rfi-Q", zeros(N))),
                              Float64.(get(c, "rfi-U", zeros(N))),
                              Float64.(get(c, "rfi-V", zeros(N))))
    GPURFISource(name, position, spectrum)
end

"""
    gpu_genvis!(vis, meta, source::GPURFISource, ra, dec, lst; position=nothing, s0=nothing)

Add the near-field model of an RFI emitter with CPU TTCal's geometry:

    ρ_i = |P - r_i| + ŝ₀·r_i,    V_ij = S(ν) exp(2πiν(ρ_j - ρ_i)/c)

with ITRF antenna positions r_i and emitter position P.  The phase centre
ŝ₀ (ITRF unit vector) is fitted from `meta.uvw` (`uvw_geometry`) unless
given: call this after `geometric_uvw!` so that ŝ₀ is the apparent phase
centre the sky models use (before it, ŝ₀ is the MS's unprecessed one).
`position` overrides P (ITRF m), e.g. to reproduce ttcal_dev, which reads
WGS84 (el, long, lat) as Cartesian.
Stokes → linear uses TTCal's convention xy = U - iV, yx = U + iV.  Works on
GPU and host visibilities (the model is built on the host).
"""
function gpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPURFISource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64;
                     position=nothing, s0=nothing)
    positions = Array(meta.antenna_positions)
    channels = Array(meta.channels)
    baselines = Array(meta.baselines)
    Na, Nb, Nf = meta.Nant, meta.Nbase, meta.Nfreq
    P = position === nothing ? source.position_itrf : SVector{3}(position)
    s0 = s0 === nothing ? uvw_geometry(meta)[1] : SVector{3}(s0)
    path = [norm(P - SVector(positions[1, i], positions[2, i], positions[3, i])) +
            (s0[1] * positions[1, i] + s0[2] * positions[2, i] + s0[3] * positions[3, i]) for i in 1:Na]
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
