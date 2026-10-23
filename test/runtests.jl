# Unit tests for TTCalX.  Run with
#     julia --project=. -t 4 test/runtests.jl
# CPU-only by default (as in CI); GPU comparisons run when CUDA is functional,
# MS I/O tests when python-casacore and the example MS are available.
using Test

const PROJECT_ROOT = dirname(@__DIR__)
include(joinpath(PROJECT_ROOT, "src", "TTCalX.jl"))
using .TTCalX

include("helpers.jl")

@testset "TTCalX" begin
    @info "Running tests" threads = Threads.nthreads() gpu = is_gpu_available()
    include("test_geometry.jl")
    include("test_rfi.jl")
    include("test_genvis_exact.jl")
    include("test_solver.jl")
    include("test_regularization.jl")
    include("test_host_backend.jl")
    include("test_ms_io.jl")
end
