# GPU Data Types for TTCal
# These types use Struct-of-Arrays layout for GPU efficiency

using CUDA
using Adapt
using StaticArrays

#==============================================================================#
#                          GPU Jones Matrix Types                               #
#==============================================================================#

"""
GPU-friendly representation of Jones matrices using Struct-of-Arrays layout.
Stores all Jones matrix components as separate arrays for coalesced memory access.

For Nant antennas and Nfreq frequencies:
- Each component array is (Nant, Nfreq)
"""
struct GPUJonesMatrixArray{T<:AbstractArray{ComplexF64}}
    xx::T
    xy::T
    yx::T
    yy::T
end

"""
GPU-friendly diagonal Jones matrices (only xx and yy components).
"""
struct GPUDiagonalJonesMatrixArray{T<:AbstractArray{ComplexF64}}
    xx::T
    yy::T
end

"""
GPU-friendly Hermitian Jones matrices.
xx and yy are real, xy is complex (yx = conj(xy)).
"""
struct GPUHermitianJonesMatrixArray{T<:AbstractArray{Float64}, S<:AbstractArray{ComplexF64}}
    xx::T  # Real
    xy::S  # Complex
    yy::T  # Real
end

# Adapt.jl integration for automatic CPU↔GPU transfers
Adapt.adapt_structure(to, x::GPUJonesMatrixArray) = 
    GPUJonesMatrixArray(adapt(to, x.xx), adapt(to, x.xy), adapt(to, x.yx), adapt(to, x.yy))

Adapt.adapt_structure(to, x::GPUDiagonalJonesMatrixArray) = 
    GPUDiagonalJonesMatrixArray(adapt(to, x.xx), adapt(to, x.yy))

Adapt.adapt_structure(to, x::GPUHermitianJonesMatrixArray) = 
    GPUHermitianJonesMatrixArray(adapt(to, x.xx), adapt(to, x.xy), adapt(to, x.yy))


#==============================================================================#
#                            GPU Visibilities                                   #
#==============================================================================#

"""
GPU-friendly visibility storage.

Shape: (Nbase, Nfreq) for each component.
"""
struct GPUVisibilities{T<:AbstractArray{ComplexF64}, F<:AbstractArray{Bool}}
    xx::T
    xy::T
    yx::T
    yy::T
    flags::F
end

function GPUVisibilities(Nbase::Int, Nfreq::Int; gpu::Bool=true)
    if gpu && CUDA.functional()
        GPUVisibilities(
            CUDA.zeros(ComplexF64, Nbase, Nfreq),
            CUDA.zeros(ComplexF64, Nbase, Nfreq),
            CUDA.zeros(ComplexF64, Nbase, Nfreq),
            CUDA.zeros(ComplexF64, Nbase, Nfreq),
            CUDA.zeros(Bool, Nbase, Nfreq)
        )
    else
        GPUVisibilities(
            zeros(ComplexF64, Nbase, Nfreq),
            zeros(ComplexF64, Nbase, Nfreq),
            zeros(ComplexF64, Nbase, Nfreq),
            zeros(ComplexF64, Nbase, Nfreq),
            zeros(Bool, Nbase, Nfreq)
        )
    end
end

Nbase(vis::GPUVisibilities) = size(vis.xx, 1)
Nfreq(vis::GPUVisibilities) = size(vis.xx, 2)

Adapt.adapt_structure(to, x::GPUVisibilities) = 
    GPUVisibilities(
        adapt(to, x.xx), adapt(to, x.xy), 
        adapt(to, x.yx), adapt(to, x.yy),
        adapt(to, x.flags)
    )


#==============================================================================#
#                            GPU Calibration                                    #
#==============================================================================#

"""
GPU-friendly calibration storage.

Shape: (Nant, Nfreq) for each component.
"""
struct GPUCalibration{T<:AbstractArray{ComplexF64}, F<:AbstractArray{Bool}}
    # For diagonal calibration: only xx, yy are used
    # For full calibration: all four are used
    xx::T
    xy::T
    yx::T
    yy::T
    flags::F
    is_diagonal::Bool
end

function GPUCalibration(Nant::Int, Nfreq::Int; diagonal::Bool=true, gpu::Bool=true)
    if gpu && CUDA.functional()
        cal = GPUCalibration(
            CUDA.ones(ComplexF64, Nant, Nfreq),
            CUDA.zeros(ComplexF64, Nant, Nfreq),
            CUDA.zeros(ComplexF64, Nant, Nfreq),
            CUDA.ones(ComplexF64, Nant, Nfreq),
            CUDA.zeros(Bool, Nant, Nfreq),
            diagonal
        )
    else
        cal = GPUCalibration(
            ones(ComplexF64, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nfreq),
            ones(ComplexF64, Nant, Nfreq),
            zeros(Bool, Nant, Nfreq),
            diagonal
        )
    end
    cal
end

Nant(cal::GPUCalibration) = size(cal.xx, 1)
Nfreq(cal::GPUCalibration) = size(cal.xx, 2)

Adapt.adapt_structure(to, x::GPUCalibration) = 
    GPUCalibration(
        adapt(to, x.xx), adapt(to, x.xy),
        adapt(to, x.yx), adapt(to, x.yy),
        adapt(to, x.flags), x.is_diagonal
    )


#==============================================================================#
#                             GPU Metadata                                      #
#==============================================================================#

"""
GPU-friendly metadata (antenna positions, baselines, frequencies, UVW).
"""
struct GPUMetadata{T<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, I<:AbstractMatrix{Int32}}
    # Antenna positions (x, y, z) in ITRF - shape (3, Nant)
    antenna_positions::T
    
    # Baseline indices - shape (2, Nbase) [antenna1, antenna2]
    baselines::I
    
    # Frequency channels - shape (Nfreq,)
    channels::V
    
    # Phase center direction (l, m, n) - shape (3,) - kept for backwards compatibility
    phase_center::V
    
    # Phase center RA/Dec in radians (J2000)
    phase_center_ra::Float64
    phase_center_dec::Float64
    
    # UVW coordinates for each baseline - shape (3, Nbase)
    uvw::T
    
    # Cached dimensions
    Nant::Int
    Nfreq::Int
    Nbase::Int
    
    # Inner constructor that computes dimensions
    function GPUMetadata(antenna_positions::T, 
                         baselines::I,
                         channels::V,
                         phase_center::V2,
                         phase_center_ra::Float64,
                         phase_center_dec::Float64,
                         uvw::T2) where {T<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, V2<:AbstractVector{Float64}, I<:AbstractMatrix{Int32}, T2<:AbstractMatrix{Float64}}
        Nant = size(antenna_positions, 2)
        Nfreq = length(channels)
        Nbase = size(baselines, 2)
        # Use the same matrix type T for uvw
        new{T, V, I}(antenna_positions, baselines, channels, convert(V, phase_center), 
                     phase_center_ra, phase_center_dec, convert(T, uvw), Nant, Nfreq, Nbase)
    end
    
    # Backwards-compatible constructor without RA/Dec (defaults to zenith)
    function GPUMetadata(antenna_positions::T, 
                         baselines::I,
                         channels::V,
                         phase_center::V2,
                         uvw::T2) where {T<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, V2<:AbstractVector{Float64}, I<:AbstractMatrix{Int32}, T2<:AbstractMatrix{Float64}}
        Nant = size(antenna_positions, 2)
        Nfreq = length(channels)
        Nbase = size(baselines, 2)
        # Default to zenith RA/Dec = (0, π/2)
        new{T, V, I}(antenna_positions, baselines, channels, convert(V, phase_center), 
                     0.0, π/2, convert(T, uvw), Nant, Nfreq, Nbase)
    end
end

# Convenience constructor with gpu flag
function create_gpu_metadata(antenna_positions::Matrix{Float64}, 
                             baselines::Matrix{Int32},
                             channels::Vector{Float64},
                             phase_center::Vector{Float64},
                             uvw::Matrix{Float64};
                             phase_center_ra::Float64=0.0,
                             phase_center_dec::Float64=π/2,
                             gpu::Bool=true)
    if gpu && CUDA.functional()
        GPUMetadata(
            CuArray(antenna_positions),
            CuArray(baselines),
            CuArray(channels),
            CuArray(phase_center),
            phase_center_ra,
            phase_center_dec,
            CuArray(uvw)
        )
    else
        GPUMetadata(
            copy(antenna_positions),
            copy(baselines),
            copy(channels),
            copy(phase_center),
            phase_center_ra,
            phase_center_dec,
            copy(uvw)
        )
    end
end

Base.getproperty(meta::GPUMetadata, s::Symbol) = begin
    if s == :Nant
        getfield(meta, :Nant)
    elseif s == :Nfreq
        getfield(meta, :Nfreq)
    elseif s == :Nbase
        getfield(meta, :Nbase)
    else
        getfield(meta, s)
    end
end

# Function methods for GPUMetadata (in addition to property access)
Nant(meta::GPUMetadata) = meta.Nant
Nfreq(meta::GPUMetadata) = meta.Nfreq
Nbase(meta::GPUMetadata) = meta.Nbase

Adapt.adapt_structure(to, x::GPUMetadata) = 
    GPUMetadata(
        adapt(to, x.antenna_positions),
        adapt(to, x.baselines),
        adapt(to, x.channels),
        adapt(to, x.phase_center),
        x.phase_center_ra,
        x.phase_center_dec,
        adapt(to, x.uvw)
    )


#==============================================================================#
#                         Square Matrix Storage                                 #
#==============================================================================#

"""
Square antenna-antenna visibility matrix for calibration.
Shape: (Nant, Nant, Nfreq) for each component.
"""
struct GPUSquareVisibilities{T<:AbstractArray{ComplexF64}}
    xx::T  # 3D array (Nant, Nant, Nfreq)
    xy::T
    yx::T
    yy::T
end

function GPUSquareVisibilities(Nant::Int, Nfreq::Int; gpu::Bool=true)
    if gpu && CUDA.functional()
        GPUSquareVisibilities(
            CUDA.zeros(ComplexF64, Nant, Nant, Nfreq),
            CUDA.zeros(ComplexF64, Nant, Nant, Nfreq),
            CUDA.zeros(ComplexF64, Nant, Nant, Nfreq),
            CUDA.zeros(ComplexF64, Nant, Nant, Nfreq)
        )
    else
        GPUSquareVisibilities(
            zeros(ComplexF64, Nant, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nant, Nfreq)
        )
    end
end

Adapt.adapt_structure(to, x::GPUSquareVisibilities) = 
    GPUSquareVisibilities(
        adapt(to, x.xx), adapt(to, x.xy),
        adapt(to, x.yx), adapt(to, x.yy)
    )
