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

    @testset "smooth_flux routing and the bright-source exemption" begin
        meta, model, measured, _ = synthetic_problem()
        ra, dec = azel_radec(80.0, 60.0)
        src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
        a, b, c = deepcopy_gpu(measured), deepcopy_gpu(measured), deepcopy_gpu(measured)
        zest_sky_rfi!(a, meta, TEST_MJD, [src]; TIGHT...)
        zest_sky_rfi!(b, meta, TEST_MJD, [src]; smooth_flux=1e5, TIGHT...)
        zest_sky_rfi!(c, meta, TEST_MJD, [src]; smooth_flux=1e5, smooth_exempt_flux=100.0, TIGHT...)
        @test a.xx != b.xx                    # smoothed: a different (band-gain) fit
        @test a.xx == c.xx && a.yy == c.yy    # catalog flux ≥ 100 Jy: exempt, per channel as before
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, [src]; smooth_flux=-1)
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, [src];
                                                 smooth_flux=10, reg=(0.5, 0.5))
    end

    @testset "warm start from returned gains" begin
        meta, src, rfi, measured = sky_rfi_problem()
        a = deepcopy_gpu(measured)
        r1 = zest_sky_rfi!(a, meta, TEST_MJD, [src], [rfi]; TIGHT...)
        @test Set(keys(r1.gains.cals)) == Set(["src", "rfi"]) && r1.gains.mjd == TEST_MJD
        b = deepcopy_gpu(measured)                      # same data, started from the solution
        r2 = zest_sky_rfi!(b, meta, TEST_MJD, [src], [rfi]; initial_gains=r1.gains, maxiter=2, peeliter=1)
        @test residual_power(b) < 1.01 * residual_power(a) + 1e-12 * residual_power(measured)
        c = deepcopy_gpu(measured)                      # cold with the same short budget is worse
        zest_sky_rfi!(c, meta, TEST_MJD, [src], [rfi]; maxiter=2, peeliter=1)
        @test residual_power(b) <= residual_power(c)
        # gains from other channels or too old are not used: same as a cold start
        for g in (merge(r1.gains, (channels=r1.gains.channels .+ 1e6,)), merge(r1.gains, (mjd=TEST_MJD - 1,)))
            d = deepcopy_gpu(measured)
            zest_sky_rfi!(d, meta, TEST_MJD, [src], [rfi]; initial_gains=g, maxiter=2, peeliter=1)
            @test d.xx == c.xx
        end
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, [src]; precision=:half)
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, [src]; smooth_exempt_flux=0)
    end

    @testset "restore_point_sources!" begin
        meta, model, measured, _ = synthetic_problem()
        ra, dec = azel_radec(80.0, 60.0)
        src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
        sm = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false); genvis_exact!(sm, meta, src, TEST_MJD)
        point(name, az, el, I, V) = (p = GPUPointSource(name, azel_radec(az, el)..., GPUPowerLaw(I, 0.0, 0.0, V, 50e6, [0.0]));
            m = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false); genvis_exact!(m, meta, p, TEST_MJD); (p, m))
        sub!(a, b, f=1.0) = (for x in (:xx, :xy, :yx, :yy); getfield(a, x) .-= f .* getfield(b, x); end; a)
        add!(a, b) = sub!(a, b, -1.0)
        jup, jm = point("jup", 250.0, 40.0, 300.0, 120.0)
        # 'peeling' took the src model and 40 % of jup: restore gives back exactly the 40 %
        before = add!(deepcopy_gpu(measured), jm)
        after = sub!(sub!(deepcopy_gpu(before), sm), jm, 0.4)
        res = restore_point_sources!(after, before, meta, TEST_MJD, [jup]; others=[sm.xx], minuvw=0.0)
        expect = sub!(add!(deepcopy_gpu(measured), jm), sm)          # before - sm
        @test residual_power(sub!(deepcopy_gpu(after), expect)) < 1e-10 * residual_power(jm)
        @test res[1][2] == (0.0, 0.0)
        @test abs(res[1][1][1][1] - 0.4 * abs(jm.xx[findfirst(!iszero, jm.xx[:, 1]), 1])) < 1e-6 * abs(jm.xx[2, 1]) + 1e-9
        # source 0.3 deg from the given position: found on Stokes V and restored there
        jup2, jm2 = point("jup2", 250.3, 40.0, 300.0, 120.0)
        before = add!(deepcopy_gpu(model), jm2)
        after = deepcopy_gpu(model)
        res = restore_point_sources!(after, before, meta, TEST_MJD, [jup]; minuvw=0.0, offset_deg=0.5)
        @test residual_power(sub!(sub!(deepcopy_gpu(after), model), jm2)) < 0.05 * residual_power(jm2)
        @test 0.1 < rad2deg(hypot(res[1][2]...)) < 0.4
        # an unpolarised source must use offset_deg = 0: then it is exact
        un, um = point("un", 250.0, 40.0, 300.0, 0.0)
        before = add!(deepcopy_gpu(model), um)
        after = deepcopy_gpu(model)
        restore_point_sources!(after, before, meta, TEST_MJD, [un]; minuvw=0.0)
        @test residual_power(sub!(sub!(deepcopy_gpu(after), model), um)) < 1e-12 * residual_power(um)
    end

    @testset "restore through zest_sky_rfi!" begin
        meta, model, measured, _ = synthetic_problem()
        ra, dec = azel_radec(80.0, 60.0)
        src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
        jup = GPUPointSource("jup", azel_radec(250.0, 40.0)..., GPUPowerLaw(300.0, 0.0, 0.0, 120.0, 50e6, [0.0]))
        jm = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false); genvis_exact!(jm, meta, jup, TEST_MJD)
        data = deepcopy_gpu(measured); gpu_putsrc!(data, jm)
        vdist(v) = sum(abs2, (v.xy .- v.yx) .- (data.xy .- data.yx))   # Stokes V vs the data (V ≈ jup)
        plain, rest = deepcopy_gpu(data), deepcopy_gpu(data)
        zest_sky_rfi!(plain, meta, TEST_MJD, [src]; TIGHT...)
        r = zest_sky_rfi!(rest, meta, TEST_MJD, [src]; restore=[jup], TIGHT...)
        @test first(r.restored[1]) == "jup"
        @test vdist(rest) < vdist(plain)
        gated = deepcopy_gpu(data)
        r = zest_sky_rfi!(gated, meta, TEST_MJD, [src]; restore=[jup], restore_below_mhz=10.0, TIGHT...)
        @test isempty(r.restored) && gated.xx == plain.xx
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(data), meta, TEST_MJD, [src];
                                                 restore=[jup], restore_offset_deg=-1)
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
            jup = GPUPointSource("jup", azel_radec(250.0, 40.0)..., GPUPowerLaw(300.0, 0.0, 0.0, 120.0, 50e6, [0.0]))
            host = deepcopy_gpu(measured)
            dev = GPUVisibilities(CuArray(measured.xx), CuArray(measured.xy), CuArray(measured.yx),
                                  CuArray(measured.yy), CuArray(measured.flags))
            kw = (smooth_flux=300.0, restore=[jup], restore_offset_deg=0.3)
            zest_sky_rfi!(host, meta, TEST_MJD, [src], [rfi]; kw...)
            zest_sky_rfi!(dev, gmeta, TEST_MJD, [src], [rfi]; kw...)
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
