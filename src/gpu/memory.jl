# Memory Management for GPU TTCal

using CUDA

#==============================================================================#
#                        CPU ↔ GPU Transfer Functions                          #
#==============================================================================#

"""
    to_gpu(x)

Transfer data structure to GPU memory.
"""
function to_gpu(x)
    if CUDA.functional()
        return adapt(CuArray, x)
    else
        return x
    end
end

"""
    to_cpu(x)

Transfer data structure back to CPU memory.
"""
function to_cpu(x)
    return adapt(Array, x)
end

"""
    to_gpu(vis::GPUVisibilities)

Ensure visibilities are on GPU.
"""
function to_gpu(vis::GPUVisibilities)
    if vis.xx isa CuArray
        return vis
    end
    GPUVisibilities(
        CuArray(vis.xx), CuArray(vis.xy),
        CuArray(vis.yx), CuArray(vis.yy),
        CuArray(vis.flags)
    )
end

"""
    to_cpu(vis::GPUVisibilities)

Transfer visibilities to CPU.
"""
function to_cpu(vis::GPUVisibilities)
    GPUVisibilities(
        Array(vis.xx), Array(vis.xy),
        Array(vis.yx), Array(vis.yy),
        Array(vis.flags)
    )
end


#==============================================================================#
#                      Conversion from Original TTCal Types                    #
#==============================================================================#

# These functions are defined in terms of the original TTCal types
# They will be used when integrating with the existing codebase

"""
    visibilities_to_gpu(data::Matrix{JonesMatrix}, flags::Matrix{Bool})

Convert CPU JonesMatrix array to GPU-friendly SoA format.
"""
function visibilities_to_gpu(data::Matrix, flags::Matrix{Bool})
    Nb, Nf = size(data)
    
    xx = zeros(ComplexF64, Nb, Nf)
    xy = zeros(ComplexF64, Nb, Nf)
    yx = zeros(ComplexF64, Nb, Nf)
    yy = zeros(ComplexF64, Nb, Nf)
    
    @inbounds for β in 1:Nf, α in 1:Nb
        J = data[α, β]
        xx[α, β] = J.xx
        xy[α, β] = J.xy
        yx[α, β] = J.yx
        yy[α, β] = J.yy
    end
    
    if CUDA.functional()
        GPUVisibilities(
            CuArray(xx), CuArray(xy),
            CuArray(yx), CuArray(yy),
            CuArray(flags)
        )
    else
        GPUVisibilities(xx, xy, yx, yy, copy(flags))
    end
end

"""
    gpu_to_visibilities(gpu_vis::GPUVisibilities)

Convert GPU visibilities back to CPU JonesMatrix array.
Returns (data, flags) tuple.
"""
function gpu_to_visibilities(gpu_vis::GPUVisibilities)
    xx = Array(gpu_vis.xx)
    xy = Array(gpu_vis.xy)
    yx = Array(gpu_vis.yx)
    yy = Array(gpu_vis.yy)
    flags = Array(gpu_vis.flags)
    
    Nb, Nf = size(xx)
    
    # Create output with proper JonesMatrix type
    # Note: This requires the JonesMatrix type from the main module
    data = Matrix{Any}(undef, Nb, Nf)
    
    @inbounds for β in 1:Nf, α in 1:Nb
        # Return as named tuple for flexibility
        data[α, β] = (xx=xx[α,β], xy=xy[α,β], yx=yx[α,β], yy=yy[α,β])
    end
    
    data, flags
end

"""
    calibration_to_gpu(jones::Matrix, flags::Matrix{Bool}; diagonal=true)

Convert CPU calibration to GPU format.
"""
function calibration_to_gpu(jones::Matrix, flags::Matrix{Bool}; diagonal::Bool=true)
    Na, Nf = size(jones)
    
    xx = zeros(ComplexF64, Na, Nf)
    xy = zeros(ComplexF64, Na, Nf)
    yx = zeros(ComplexF64, Na, Nf)
    yy = zeros(ComplexF64, Na, Nf)
    
    @inbounds for β in 1:Nf, α in 1:Na
        J = jones[α, β]
        if diagonal
            # DiagonalJonesMatrix only has xx and yy
            xx[α, β] = J.xx
            yy[α, β] = J.yy
        else
            # Full JonesMatrix has all four
            xx[α, β] = J.xx
            xy[α, β] = J.xy
            yx[α, β] = J.yx
            yy[α, β] = J.yy
        end
    end
    
    if CUDA.functional()
        GPUCalibration(
            CuArray(xx), CuArray(xy),
            CuArray(yx), CuArray(yy),
            CuArray(flags), diagonal
        )
    else
        GPUCalibration(xx, xy, yx, yy, copy(flags), diagonal)
    end
end


#==============================================================================#
#                          Memory Pool Management                              #
#==============================================================================#

"""
Preallocated memory pools for GPU operations.
Reduces allocation overhead during iterative algorithms.
"""
mutable struct GPUMemoryPool
    # Work arrays for stefcal
    square_measured::Union{GPUSquareVisibilities, Nothing}
    square_model::Union{GPUSquareVisibilities, Nothing}
    step_buffer::Union{GPUCalibration, Nothing}
    
    # Dimensions
    Nant::Int
    Nbase::Int
    Nfreq::Int
    
    function GPUMemoryPool(Nant::Int, Nbase::Int, Nfreq::Int; gpu::Bool=true)
        new(nothing, nothing, nothing, Nant, Nbase, Nfreq)
    end
end

"""
    allocate!(pool::GPUMemoryPool; diagonal=true)

Allocate all work arrays in the pool.
"""
function allocate!(pool::GPUMemoryPool; diagonal::Bool=true, gpu::Bool=true)
    pool.square_measured = GPUSquareVisibilities(pool.Nant, pool.Nfreq; gpu=gpu)
    pool.square_model = GPUSquareVisibilities(pool.Nant, pool.Nfreq; gpu=gpu)
    pool.step_buffer = GPUCalibration(pool.Nant, pool.Nfreq; diagonal=diagonal, gpu=gpu)
    pool
end

"""
    free!(pool::GPUMemoryPool)

Free all GPU memory in the pool.
"""
function free!(pool::GPUMemoryPool)
    pool.square_measured = nothing
    pool.square_model = nothing
    pool.step_buffer = nothing
    CUDA.functional() && CUDA.reclaim()
    pool
end

"""
    gpu_memory_stats()

Print current GPU memory statistics.
"""
function gpu_memory_stats()
    if CUDA.functional()
        total = CUDA.total_memory()
        free = CUDA.available_memory()
        used = total - free
        println("GPU Memory:")
        println("  Total: $(round(total/1e9, digits=2)) GB")
        println("  Used:  $(round(used/1e9, digits=2)) GB")
        println("  Free:  $(round(free/1e9, digits=2)) GB")
    else
        println("CUDA not available")
    end
end
