# GPU-accelerated TTCal Module
# 
# High-performance calibration for radio interferometry using CUDA.
# Implements peeling/zesting for direction-dependent calibration.
#
# Copyright (c) 2024
# License: GPL v3

module GPUTTCal

using CUDA
using KernelAbstractions
using Adapt
using StaticArrays
using LinearAlgebra
using JSON
using Printf

# Re-export CUDA utilities
export CuArray, CUDA

# Export GPU types
export GPUJonesMatrixArray, GPUDiagonalJonesMatrixArray, GPUHermitianJonesMatrixArray
export GPUVisibilities, GPUCalibration, GPUSquareVisibilities
export GPUMetadata, create_gpu_metadata

# Export source types
export GPUSource, GPUPointSource, GPUGaussianSource, GPUMultiSource
export GPUPowerLaw, read_gpu_sources
export AbstractGPUPeelingSource, GPUPeelingSource, GPUShavingSource, GPUZestingSource, GPUPruningSource
export unwrap, get_name, is_diagonal, is_wideband, source_direction_lmn, calibration_type

# Export GPU operations
export gpu_corrupt!, gpu_applycal!
export gpu_stefcal!, gpu_stefcal_step_diagonal!, gpu_stefcal_step_full!
export gpu_genvis!, gpu_genvis, gpu_makesquare!
export gpu_subsrc!, gpu_putsrc!

# Export peel/zest operations
export peel_gpu!, shave_gpu!, zest_gpu!, prune_gpu!

# Export utilities  
export to_gpu, to_cpu, deepcopy_gpu
export is_gpu_available, get_gpu_backend
export Nbase, Nfreq, Nant

# Export MS bridge functions (require python-casacore via PyCall at runtime)
export read_ms_to_gpu, write_gpu_to_ms!
export init_pycasacore

# Export logging utilities
export set_verbosity, get_verbosity, is_quiet, is_verbose, is_normal
export ProgressBar, update!, finish!
export log_header, log_section, log_step, log_substep, log_detail, log_debug
export log_success, log_warning, log_error, log_config, log_table_row
export @verbose, @normal

# Include sub-modules
include("logging.jl")
include("types.jl")
include("memory.jl")
include("kernels/utils.jl")
include("kernels/corrupt.jl")
include("kernels/stefcal.jl")
include("kernels/genvis.jl")
include("sources.jl")
include("peel_gpu.jl")
include("pycall_ms_bridge.jl")

"""
    is_gpu_available()

Check if CUDA GPU is available and functional.
"""
function is_gpu_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end

"""
    get_gpu_backend()

Get the current GPU backend: :cuda or :cpu
"""
function get_gpu_backend()
    if CUDA.functional()
        return :cuda
    else
        return :cpu
    end
end

function __init__()
    if is_gpu_available()
        @info "GPUTTCal initialized with CUDA" device=CUDA.name(CUDA.device()) memory="$(round(CUDA.available_memory()/1e9, digits=2)) GB"
    else
        @warn "CUDA not available - using CPU fallback"
    end
end

end # module
