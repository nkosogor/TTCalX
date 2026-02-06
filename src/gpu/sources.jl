# GPU Source Models
# Source representations and JSON loading for GPU-accelerated peeling/zesting

using JSON
using StaticArrays

#==============================================================================#
#                           Spectrum Types                                      #
#==============================================================================#

"""
Power-law spectrum for astronomical sources.
flux(ν) = stokes * (ν / reference_frequency)^index[1] * ...
"""
struct GPUPowerLaw
    I::Float64
    Q::Float64
    U::Float64
    V::Float64
    reference_frequency::Float64
    index::Vector{Float64}
end

function (spectrum::GPUPowerLaw)(frequency::Float64)
    ratio = log(frequency / spectrum.reference_frequency)
    log_flux = zero(Float64)
    for (i, idx) in enumerate(spectrum.index)
        log_flux += idx * ratio^i
    end
    flux_scale = exp(log_flux)
    return (spectrum.I * flux_scale, spectrum.Q * flux_scale, 
            spectrum.U * flux_scale, spectrum.V * flux_scale)
end

#==============================================================================#
#                            Source Types                                       #
#==============================================================================#

abstract type GPUSource end

"""Point source - astronomical point source."""
struct GPUPointSource <: GPUSource
    name::String
    ra::Float64   # Right ascension in radians
    dec::Float64  # Declination in radians
    spectrum::GPUPowerLaw
end

"""Gaussian source - elliptical Gaussian shape."""
struct GPUGaussianSource <: GPUSource
    name::String
    ra::Float64
    dec::Float64
    spectrum::GPUPowerLaw
    major_fwhm::Float64      # FWHM along major axis (radians)
    minor_fwhm::Float64      # FWHM along minor axis (radians)
    position_angle::Float64  # Position angle (radians)
end

"""Multi-component source."""
struct GPUMultiSource <: GPUSource
    name::String
    components::Vector{GPUSource}
end

#==============================================================================#
#                        Peeling Source Wrappers                                #
#==============================================================================#

"""Wrapper that tells peeling which calibration type to use."""
abstract type AbstractGPUPeelingSource end

"""Peeling: diagonal Jones, per frequency channel."""
struct GPUPeelingSource <: AbstractGPUPeelingSource
    source::GPUSource
end

"""Shaving: diagonal Jones, one per subband."""
struct GPUShavingSource <: AbstractGPUPeelingSource
    source::GPUSource
end

"""Zesting: full Jones, per frequency channel."""
struct GPUZestingSource <: AbstractGPUPeelingSource
    source::GPUSource
end

"""Pruning: full Jones, one per subband."""
struct GPUPruningSource <: AbstractGPUPeelingSource
    source::GPUSource
end

unwrap(s::AbstractGPUPeelingSource) = s.source
unwrap(s::GPUSource) = s

# Create appropriate calibration type for each peeling mode
function calibration_type(::GPUPeelingSource, Nant::Int, Nfreq::Int; gpu::Bool=true)
    GPUCalibration(Nant, Nfreq, diagonal=true, gpu=gpu)
end

function calibration_type(::GPUShavingSource, Nant::Int, Nfreq::Int; gpu::Bool=true)
    GPUCalibration(Nant, 1, diagonal=true, gpu=gpu)
end

function calibration_type(::GPUZestingSource, Nant::Int, Nfreq::Int; gpu::Bool=true)
    GPUCalibration(Nant, Nfreq, diagonal=false, gpu=gpu)
end

function calibration_type(::GPUPruningSource, Nant::Int, Nfreq::Int; gpu::Bool=true)
    GPUCalibration(Nant, 1, diagonal=false, gpu=gpu)
end

is_diagonal(::GPUPeelingSource) = true
is_diagonal(::GPUShavingSource) = true
is_diagonal(::GPUZestingSource) = false
is_diagonal(::GPUPruningSource) = false

is_wideband(::GPUPeelingSource) = false
is_wideband(::GPUShavingSource) = true
is_wideband(::GPUZestingSource) = false
is_wideband(::GPUPruningSource) = true

#==============================================================================#
#                          JSON Loading                                         #
#==============================================================================#

"""
    read_gpu_sources(filename::String) -> Vector{GPUSource}

Read sources from a JSON file in standard TTCal format.
"""
function read_gpu_sources(filename::String)
    data = JSON.parsefile(filename)
    sources = GPUSource[]
    for c in data
        # Convert JSON.Object to Dict if needed
        source = construct_gpu_source(Dict(c))
        push!(sources, source)
    end
    return sources
end

function construct_gpu_source(c::Dict)
    name = get(c, "name", "")
    
    if haskey(c, "components")
        # Multi-component source - convert each component
        components = GPUSource[]
        for d in c["components"]
            push!(components, construct_gpu_source(Dict(d)))
        end
        return GPUMultiSource(name, components)
    end
    
    # Parse direction
    ra, dec = parse_source_direction(c)
    
    # Parse spectrum
    spectrum = parse_source_spectrum(c)
    
    if haskey(c, "major-fwhm") && haskey(c, "minor-fwhm") && haskey(c, "position-angle")
        # Gaussian source
        major_fwhm = deg2rad(c["major-fwhm"] / 3600)  # arcsec to radians
        minor_fwhm = deg2rad(c["minor-fwhm"] / 3600)
        position_angle = deg2rad(c["position-angle"])
        return GPUGaussianSource(name, ra, dec, spectrum, major_fwhm, minor_fwhm, position_angle)
    else
        # Point source
        return GPUPointSource(name, ra, dec, spectrum)
    end
end

function parse_source_direction(c::Dict)
    name = get(c, "name", "")
    
    # Handle special names
    if name == "Sun" || name == "Moon" || name == "Jupiter"
        @warn "Moving sources (Sun/Moon/Jupiter) require runtime coordinate conversion - using placeholder"
        return (0.0, 0.0)
    end
    
    if haskey(c, "ra") && haskey(c, "dec")
        ra = parse_angle(c["ra"])
        dec = parse_angle(c["dec"])
        return (ra, dec)
    elseif haskey(c, "az") && haskey(c, "el")
        @warn "AZ/EL coordinates require runtime conversion - using placeholder"
        return (0.0, 0.0)
    else
        error("Source $name has no valid coordinates (need ra/dec or az/el)")
    end
end

"""Parse angle from string like "19h59m28.35663s" or "+40d44m02.0970s" or numeric (radians)."""
function parse_angle(s)
    if s isa Number
        return Float64(s)
    end
    
    s = String(s)
    
    # Hour angle format: 19h59m28.35663s
    m = match(r"([+-]?\d+(?:\.\d+)?)[hH](\d+(?:\.\d+)?)[mM](\d+(?:\.\d+)?)[sS]?", s)
    if m !== nothing
        h = parse(Float64, m.captures[1])
        mins = parse(Float64, m.captures[2])
        secs = parse(Float64, m.captures[3])
        sign_val = h >= 0 ? 1 : -1
        return sign_val * (abs(h) + mins/60 + secs/3600) * (π / 12)  # hours to radians
    end
    
    # Degree format: +40d44m02.0970s
    m = match(r"([+-]?\d+(?:\.\d+)?)[dD](\d+(?:\.\d+)?)[mM']?(\d+(?:\.\d+)?)[sS\"]?", s)
    if m !== nothing
        d = parse(Float64, m.captures[1])
        mins = parse(Float64, m.captures[2])
        secs = parse(Float64, m.captures[3])
        sign_val = startswith(strip(s), "-") ? -1 : 1
        return sign_val * (abs(d) + mins/60 + secs/3600) * (π / 180)  # degrees to radians
    end
    
    # Simple numeric degrees
    m = match(r"([+-]?\d+(?:\.\d+)?)", s)
    if m !== nothing
        return parse(Float64, m.captures[1]) * (π / 180)
    end
    
    error("Cannot parse angle: $s")
end

function parse_source_spectrum(c::Dict)
    I = Float64(c["I"])
    Q = Float64(get(c, "Q", 0.0))
    U = Float64(get(c, "U", 0.0))
    V = Float64(get(c, "V", 0.0))
    freq = Float64(c["freq"])
    index = Float64.(c["index"])
    return GPUPowerLaw(I, Q, U, V, freq, index)
end

#==============================================================================#
#                      Coordinate Conversions                                   #
#==============================================================================#

"""Convert J2000 RA/Dec to ITRF direction for a given time and position."""
function j2000_to_itrf(ra::Float64, dec::Float64, lst::Float64)
    # Local hour angle
    ha = lst - ra
    
    # Direction cosines in topocentric coordinates
    l = cos(dec) * sin(ha)
    m = cos(dec) * cos(ha)
    n = sin(dec)
    
    return SVector(l, m, n)
end

"""Get source direction relative to phase center."""
function source_direction_lmn(source::GPUSource, phase_center_ra::Float64, 
                               phase_center_dec::Float64, lst::Float64)
    ra = source.ra
    dec = source.dec
    
    # Direction cosines relative to phase center
    Δra = ra - phase_center_ra
    
    l = cos(dec) * sin(Δra)
    m = sin(dec) * cos(phase_center_dec) - cos(dec) * sin(phase_center_dec) * cos(Δra)
    n = sin(dec) * sin(phase_center_dec) + cos(dec) * cos(phase_center_dec) * cos(Δra)
    
    return SVector(l, m, n)  # Return full n, kernel handles (n-1)
end

function source_direction_lmn(source::GPUGaussianSource, phase_center_ra::Float64,
                               phase_center_dec::Float64, lst::Float64)
    ra = source.ra
    dec = source.dec
    
    Δra = ra - phase_center_ra
    
    l = cos(dec) * sin(Δra)
    m = sin(dec) * cos(phase_center_dec) - cos(dec) * sin(phase_center_dec) * cos(Δra)
    n = sin(dec) * sin(phase_center_dec) + cos(dec) * cos(phase_center_dec) * cos(Δra)
    
    return SVector(l, m, n)  # Return full n, kernel handles (n-1)
end

function source_direction_lmn(source::GPUMultiSource, phase_center_ra::Float64, 
                               phase_center_dec::Float64, lst::Float64)
    # Use first component direction
    if !isempty(source.components)
        return source_direction_lmn(source.components[1], phase_center_ra, phase_center_dec, lst)
    end
    return SVector(0.0, 0.0, 1.0)
end

#==============================================================================#
#                           Source Queries                                      #
#==============================================================================#

"""Check if source is above horizon (elevation > 0)."""
function is_above_horizon(source::GPUSource, phase_center_ra::Float64, 
                          phase_center_dec::Float64, lst::Float64)
    # Simple check: n > 0 in source direction
    lmn = source_direction_lmn(source, phase_center_ra, phase_center_dec, lst)
    return lmn[3] > 0.0  # n > 0 means source is above horizon relative to phase center
end

"""Get source name."""
get_name(s::GPUSource) = s.name
get_name(s::AbstractGPUPeelingSource) = get_name(unwrap(s))

"""Get number of sources."""
count_sources(sources::Vector{<:GPUSource}) = length(sources)
count_sources(sources::Vector{<:AbstractGPUPeelingSource}) = length(sources)
