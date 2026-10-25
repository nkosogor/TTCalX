# zest_sky_rfi!: sky + RFI peeling of one dataset in one call.
# Uses synthetic_problem (test_solver.jl) and casacore_available (test_ms_io.jl).

"""RFI emitter 20 km due north, on the ground, with a flat spectrum."""
function ground_rfi(I=300.0)
    _, north, _ = enu_basis()
    GPURFISource("rfi", SVector{3}(Vector(OVRO_ITRF) .+ 2e4 .* north),
                 GPURFISpectrum([1e6, 1e9], [I, I], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]))
end

"""Data = sky source with random gains + RFI emitter with other random gains."""
function sky_rfi_problem()
    meta, sky_model, measured, _ = synthetic_problem()
    ra, dec = azel_radec(80.0, 60.0)
    src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
    rfi = ground_rfi()
    rfi_model = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
    gpu_genvis!(rfi_model, meta, rfi, meta.phase_center_ra, meta.phase_center_dec, 0.0)
    gpu_corrupt!(rfi_model, random_gains(meta.Nant, meta.Nfreq; seed=9, ε=0.05), meta)
    gpu_putsrc!(measured, rfi_model)
    meta, src, rfi, measured
end

const TIGHT = (maxiter=30, tolerance=1e-8)

@testset "zest_sky_rfi!" begin
    @testset "sky only, below-horizon source skipped" begin
        meta, model, measured, _ = synthetic_problem()
        ra, dec = azel_radec(80.0, 60.0)
        src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
        rl, dl = azel_radec(200.0, -5.0)
        low = GPUPointSource("low", rl, dl, GPUPowerLaw(500.0, 0.0, 0.0, 0.0, 50e6, [-0.7]))
        @test source_elevation_deg(src, meta) ≈ 60 atol = 1
        @test isnan(source_elevation_deg(ground_rfi(), meta))
        vis = deepcopy_gpu(measured)
        r = zest_sky_rfi!(vis, meta, TEST_MJD, [src, low]; TIGHT...)
        @test r.peeled == ["src"] && r.skipped == ["low"] && r.flagged_zeros == 0
        @test residual_power(vis) < 1e-6 * residual_power(measured)
    end

    @testset "RFI after the sky and jointly" begin
        meta, src, rfi, measured = sky_rfi_problem()
        p0 = residual_power(measured)
        for mode in (:after, :joint)
            vis = deepcopy_gpu(measured)
            r = zest_sky_rfi!(vis, meta, TEST_MJD, [src], [rfi]; rfi_mode=mode, TIGHT...)
            @test Set(r.peeled) == Set(["src", "rfi"])
            @test residual_power(vis) < 1e-3 * p0      # :after solves the sky with the RFI still in
        end
        # Sky only leaves the RFI in
        vis = deepcopy_gpu(measured)
        zest_sky_rfi!(vis, meta, TEST_MJD, [src]; TIGHT...)
        @test residual_power(vis) > 1e-2 * p0
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, [src], [rfi];
                                                 rfi_mode=:before)
    end

    @testset "solver, peel, regularisation, zero flags, report" begin
        meta, model, measured, _ = synthetic_problem()
        ra, dec = azel_radec(80.0, 60.0)
        src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
        p0 = residual_power(measured)
        vis = deepcopy_gpu(measured)
        zest_sky_rfi!(vis, meta, TEST_MJD, [src]; solver=:damped, maxiter=100, tolerance=1e-8)
        @test residual_power(vis) < 1e-4 * p0
        vis = deepcopy_gpu(measured)                    # diagonal gains cannot fit the leakage
        r = zest_sky_rfi!(vis, meta, TEST_MJD, [src]; diagonal=true, TIGHT...)
        @test r.peeled == ["src"] && residual_power(vis) < 0.1 * p0
        vis = deepcopy_gpu(measured)
        zest_sky_rfi!(vis, meta, TEST_MJD, [src]; reg=(0.5, 0.5, 1.0), TIGHT...)
        @test residual_power(vis) < 1e-2 * p0
        vis = deepcopy_gpu(measured)
        vis.xx[3, :] .= 0; vis.yy[3, :] .= 0
        log = []
        r = zest_sky_rfi!(vis, meta, TEST_MJD, [src]; peeliter=2, TIGHT...,
                          report=(name, it; action, kw...) -> push!(log, (name, it, action)))
        @test r.flagged_zeros == meta.Nfreq && all(vis.flags[3, :])
        @test [(l[1], l[2]) for l in log] == [("src", 1), ("src", 2)] && log[1][3] == :accept
    end

    if is_gpu_available()
        @testset "GPU equals host" begin
            meta, src, rfi, measured = sky_rfi_problem()
            gmeta = synthetic_meta(gpu=true)
            host = deepcopy_gpu(measured)
            dev = GPUVisibilities(CuArray(measured.xx), CuArray(measured.xy), CuArray(measured.yx),
                                  CuArray(measured.yy), CuArray(measured.flags))
            zest_sky_rfi!(host, meta, TEST_MJD, [src], [rfi])
            zest_sky_rfi!(dev, gmeta, TEST_MJD, [src], [rfi])
            @test relmax(host_copy(dev), host) < 1e-8
        end
    end

    ms_src = joinpath(dirname(@__DIR__), "examples", "20240524_090003_73MHz_ch0to5.ms")
    if casacore_available() && isdir(ms_src)
        @testset "zest_sky_rfi_ms! on the example MS" begin
            dir = mktempdir()
            ms = joinpath(dir, "test.ms")
            cp(ms_src, ms)
            sky = read_gpu_sources(joinpath(dirname(@__DIR__), "sources.json"))
            before, = read_ms_fast(ms; gpu=false, column="DATA")
            r = zest_sky_rfi_ms!(ms, sky; column="DATA", gpu=false)
            after, = read_ms_fast(ms; gpu=false, column="DATA")
            @test !isempty(r.peeled)
            @test residual_power(after) < residual_power(before)
            rm(dir; recursive=true)
        end
    end
end
