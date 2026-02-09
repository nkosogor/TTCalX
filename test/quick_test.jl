#!/usr/bin/env julia
#==============================================================================#
#               Quick Test — CPU-only smoke test for CI                         #
#==============================================================================#
#
# Runs a minimal zest pipeline on the example 6-channel MS using CPU fallback.
# This script is designed to run in GitHub Actions where CUDA is not available.
#
# Usage:
#   julia --project=. test/quick_test.jl
#
# Requirements:
#   - python-casacore (pip install python-casacore)
#   - PyCall configured with the same Python
#   - Example MS in examples/
#
#==============================================================================#

using Test

# ─── Setup paths ────────────────────────────────────────────────────────────
const PROJECT_ROOT = dirname(@__DIR__)
const EXAMPLE_MS   = joinpath(PROJECT_ROOT, "examples", "20240524_090003_73MHz_ch0to5.ms")
const SOURCES_JSON = joinpath(PROJECT_ROOT, "sources.json")

# Work on a copy so we never modify the repo's example data
const WORK_DIR = mktempdir()
const TEST_MS  = joinpath(WORK_DIR, "test_quick.ms")

println("="^72)
println("  TTCalX Quick Test (CPU fallback)")
println("="^72)
println()

# ─── 1. Load TTCalX ─────────────────────────────────────────────────────────
@testset "Module loading" begin
    push!(LOAD_PATH, joinpath(PROJECT_ROOT, "src"))
    include(joinpath(PROJECT_ROOT, "src", "TTCalX.jl"))
    using .TTCalX

    @test !is_gpu_available()  || @info "GPU is available (unexpected in CI, but OK)"
    backend = get_gpu_backend()
    @test backend in (:cpu, :cuda)
    println("  Backend: $backend")
end

# ─── 2. Copy example MS to temp directory ────────────────────────────────────
@testset "Copy example MS" begin
    @assert isdir(EXAMPLE_MS) "Example MS not found at $EXAMPLE_MS"
    cp(EXAMPLE_MS, TEST_MS)
    @test isdir(TEST_MS)
    println("  Copied to: $TEST_MS")
end

# ─── 3. Initialize python-casacore ──────────────────────────────────────────
HAS_PYCASACORE = false
@testset "python-casacore" begin
    global HAS_PYCASACORE
    ok = TTCalX.init_pycasacore()
    HAS_PYCASACORE = ok
    if !ok
        @warn "python-casacore not available — MS I/O tests will be skipped" *
              "\n  Install with: pip install python-casacore" *
              "\n  (On macOS, use conda: conda install -c conda-forge python-casacore)"
        # In CI (GitHub Actions), this MUST succeed — fail hard
        if get(ENV, "CI", "") == "true"
            @test ok
        else
            @test_skip ok
        end
    else
        @test ok
        println("  python-casacore initialized")
    end
end

# ─── 4. Read sources (pure Julia — always works) ────────────────────────────
local sources
@testset "Read sources" begin
    @assert isfile(SOURCES_JSON) "sources.json not found at $SOURCES_JSON"
    sources = TTCalX.read_gpu_sources(SOURCES_JSON)
    @test length(sources) > 0
    println("  Loaded $(length(sources)) sources:")
    for s in sources
        println("    - $(TTCalX.get_name(s))")
    end
end

if !HAS_PYCASACORE
    println()
    println("⚠  Skipping MS I/O and zest tests (python-casacore not available)")
    println("   The following tests require python-casacore and will run in CI.")
    println()
else

# ─── 5. Read MS on CPU ──────────────────────────────────────────────────────
local vis, cal, meta, baseline_dict, Nrows
@testset "Read MS (CPU)" begin
    vis, cal, meta, baseline_dict, Nrows = TTCalX.read_ms_to_gpu(
        TEST_MS; gpu=false, column="DATA"
    )

    Nant  = TTCalX.Nant(meta)
    Nbase = TTCalX.Nbase(meta)
    Nfreq = TTCalX.Nfreq(meta)

    @test Nant  > 0
    @test Nbase > 0
    @test Nfreq == 6  # example MS has 6 channels

    println("  Nant=$Nant  Nbase=$Nbase  Nfreq=$Nfreq  Nrows=$Nrows")

    # Visibilities should have data
    @test !all(iszero, vis.xx)
    @test size(vis.xx) == (Nbase, Nfreq)
end

# ─── 6. Run zest on CPU ─────────────────────────────────────────────────────
@testset "Zest (CPU)" begin
    # Save a copy of original visibilities to verify they changed
    orig_xx = copy(vis.xx)

    println("  Running zest (CPU)...")
    t0 = time()

    TTCalX.zest_gpu!(vis, meta, sources;
        maxiter   = 5,       # few iterations for speed
        tolerance = 1e-3,
        minuvw    = 10.0,
        peeliter  = 1,       # single pass for speed
        phase_center_ra  = meta.phase_center_ra,
        phase_center_dec = meta.phase_center_dec,
        lst = 0.0,
    )

    elapsed = time() - t0
    println("  Zest completed in $(round(elapsed, digits=2)) seconds")

    # Visibilities should have changed after peeling
    @test vis.xx != orig_xx
    # No NaN/Inf should be introduced
    @test !any(isnan, vis.xx)
    @test !any(isinf, vis.xx)
    println("  ✓ Visibilities modified, no NaN/Inf")
end

# ─── 7. Write back to MS ────────────────────────────────────────────────────
@testset "Write MS" begin
    TTCalX.write_gpu_to_ms!(TEST_MS, vis, baseline_dict, Nrows; column="DATA")
    println("  ✓ Wrote calibrated data back to MS")
end

end  # if HAS_PYCASACORE

# ─── Cleanup ─────────────────────────────────────────────────────────────────
rm(WORK_DIR; recursive=true, force=true)

println()
println("="^72)
println("  All quick tests passed ✓")
println("="^72)
