"""Synthetic zest problem: model of a bright source, data = J M Jᴴ (+ noise) with known gains."""
function synthetic_problem(; noise=0.0, seed=4, Nant=24, channels=[40e6, 55e6, 70e6])
    meta = synthetic_meta(; Nant=Nant, channels=channels)
    ra, dec = azel_radec(80.0, 60.0)
    src = GPUPointSource("src", ra, dec, GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
    model = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
    genvis_exact!(model, meta, src, TEST_MJD)
    truth = random_gains(meta.Nant, meta.Nfreq; seed=seed)
    measured = deepcopy_gpu(model)
    gpu_corrupt!(measured, truth, meta)
    if noise > 0
        rng = MersenneTwister(seed + 1)
        for f in (:xx, :xy, :yx, :yy)
            getfield(measured, f) .+= noise .* randn(rng, ComplexF64, meta.Nbase, meta.Nfreq)
        end
    end
    meta, model, measured, truth
end

identity_gains(meta; diagonal=false) = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=diagonal, gpu=false)

@testset "Solver" begin
    @testset "mask_short_baselines! matches flag_short_baselines!" begin
        meta = synthetic_meta()
        vis = random_vis(meta.Nbase, meta.Nfreq; flag_fraction=0.1)
        a = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        host_makesquare!(a, vis, meta)
        b = deepcopy(a)
        minuvw = 40.0          # long enough to cut part of the synthetic array
        mask_short_baselines!(a, meta, minuvw)
        TTCalX.flag_short_baselines!(b, meta, minuvw)
        @test a.xx == b.xx && a.xy == b.xy && a.yx == b.yx && a.yy == b.yy
        @test count(iszero, a.xx) > count(iszero, (host_makesquare!(GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false), vis, meta)).xx)
    end

    @testset "host_makesquare!: Hermitian, autos and flags zero" begin
        meta = synthetic_meta()
        vis = random_vis(meta.Nbase, meta.Nfreq; flag_fraction=0.2)
        sq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        host_makesquare!(sq, vis, meta)
        bl = meta.baselines
        for β in 1:meta.Nfreq
            @test all(iszero, [sq.xx[i, i, β] for i in 1:meta.Nant])
            @test sq.xx[:, :, β] == sq.xx[:, :, β]'
            @test sq.xy[:, :, β] == sq.yx[:, :, β]'
            for α in 1:meta.Nbase
                a1, a2 = bl[1, α], bl[2, α]
                expected = vis.flags[α, β] ? zero(ComplexF64) : vis.xy[α, β]
                @test sq.xy[a1, a2, β] == expected
            end
        end
    end

    @testset "full-Jones step matches TTCalX's CPU step" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        gains = random_gains(meta.Nant, meta.Nfreq; seed=9, ε=0.2)
        msq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        Msq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        host_makesquare!(msq, measured, meta)
        host_makesquare!(Msq, model, meta)
        g = (gains.xx, gains.xy, gains.yx, gains.yy)
        k = ntuple(_ -> zeros(ComplexF64, meta.Nant, meta.Nfreq), 4)
        stefcal_steps!(k, g, false, msq, Msq, meta.Nant)
        r = ntuple(_ -> zeros(ComplexF64, meta.Nant, meta.Nfreq), 4)
        TTCalX.cpu_stefcal_step_full!(r..., gains, msq, Msq, meta)
        @test maximum(maximum(abs, k[c] .- r[c]) for c in 1:4) < 1e-12 * maximum(maximum(abs, r[c]) for c in 1:4)
    end

    @testset "diagonal step matches TTCalX's CPU step" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        gains = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=true, gpu=false)
        gains.xx .*= cis.(0.1 .* randn(MersenneTwister(5), meta.Nant, meta.Nfreq))
        msq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        Msq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
        host_makesquare!(msq, measured, meta)
        host_makesquare!(Msq, model, meta)
        k = ntuple(_ -> zeros(ComplexF64, meta.Nant, meta.Nfreq), 4)
        stefcal_steps!(k, (gains.xx, gains.xy, gains.yx, gains.yy), true, msq, Msq, meta.Nant)
        rxx, ryy = zeros(ComplexF64, meta.Nant, meta.Nfreq), zeros(ComplexF64, meta.Nant, meta.Nfreq)
        TTCalX.cpu_stefcal_step_diagonal!(rxx, ryy, gains, msq, Msq, meta)
        @test k[1] ≈ rxx rtol = 1e-12
        @test k[4] ≈ ryy rtol = 1e-12
    end

    @testset "solve_gains! fits noiseless data ($solver)" for solver in (:rk4, :damped)
        meta, model, measured, _ = synthetic_problem()
        cal = identity_gains(meta)
        maxiter = solver == :rk4 ? 30 : 300
        converged, niters, frac = solve_gains!(cal, measured, model, meta; solver=solver, maxiter=maxiter, tolerance=1e-10)
        @test converged && frac == 1.0 && niters <= maxiter
        @test fit_residual(measured, model, cal, meta) < 1e-8
    end

    @testset "solve_gains! with a diagonal calibration ($solver)" for solver in (:rk4, :damped)
        # Unpolarized cross-hands: with diagonal gains the X-Y phase difference is fixed
        # only through the XY/YX model, and a 1%-polarized model makes that direction
        # converge very slowly (as in CPU TTCal's diagonal StefCal).
        meta, model, _, _ = synthetic_problem()
        model.xy .= 0
        model.yx .= 0
        truth = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=true, gpu=false)
        rng = MersenneTwister(21)
        truth.xx .= (1 .+ 0.1 .* randn(rng, meta.Nant, meta.Nfreq)) .* cis.(0.3 .* randn(rng, meta.Nant, meta.Nfreq))
        truth.yy .= (1 .+ 0.1 .* randn(rng, meta.Nant, meta.Nfreq)) .* cis.(0.3 .* randn(rng, meta.Nant, meta.Nfreq))
        measured = deepcopy_gpu(model)
        gpu_corrupt!(measured, truth, meta)
        cal = identity_gains(meta; diagonal=true)
        maxiter = solver == :rk4 ? 30 : 300
        converged, _, _ = solve_gains!(cal, measured, model, meta; solver=solver, maxiter=maxiter, tolerance=1e-10)
        @test converged
        @test fit_residual(measured, model, cal, meta) < 1e-8
        @test all(iszero, cal.xy) && all(iszero, cal.yx)
    end

    @testset "solve_gains! with noise reaches the noise floor" begin
        meta, model, measured, truth = synthetic_problem(noise=5.0)
        cal = identity_gains(meta)
        solve_gains!(cal, measured, model, meta; maxiter=30, tolerance=1e-8)
        best = fit_residual(measured, model, truth, meta)       # residual with the true gains
        @test fit_residual(measured, model, cal, meta) <= best * (1 + 1e-6)
    end

    @testset "CPU TTCal settings: per-channel stopping and iteration count" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        cal = identity_gains(meta)
        converged, niters, frac = solve_gains!(cal, measured, model, meta; maxiter=1, tolerance=1e-4)
        @test niters == 1 && !converged && frac == 0.0
        cal = identity_gains(meta)
        converged, niters, frac = solve_gains!(cal, measured, model, meta; maxiter=5, tolerance=1e-4, minuvw=5.0)
        @test niters <= 5 && 0.0 <= frac <= 1.0 && converged == (frac == 1.0)
    end

    @testset "minuvw keeps corrupted short baselines out of the solve" begin
        meta, model, measured, _ = synthetic_problem()
        pos, bl = meta.antenna_positions, meta.baselines
        minuvw = 40.0          # 171 m at 70 MHz: ~5% of the synthetic baselines
        short = [norm(pos[:, bl[2, α]] .- pos[:, bl[1, α]]) < minuvw * SPEED_OF_LIGHT / maximum(meta.channels)
                 for α in 1:meta.Nbase]
        @test any(short)
        bad = deepcopy_gpu(measured)
        bad.xx[short, :] .+= 1e5
        cal = identity_gains(meta)
        solve_gains!(cal, bad, model, meta; maxiter=30, tolerance=1e-10, minuvw=minuvw)
        keep = .!short
        sub = GPUVisibilities(measured.xx[keep, :], measured.xy[keep, :], measured.yx[keep, :],
                              measured.yy[keep, :], measured.flags[keep, :])
        msub = GPUVisibilities(model.xx[keep, :], model.xy[keep, :], model.yx[keep, :], model.yy[keep, :], model.flags[keep, :])
        metasub = create_gpu_metadata(Array(meta.antenna_positions), Array(meta.baselines[:, keep]), Array(meta.channels),
                                      [0.0, 0.0, 1.0], Array(meta.uvw[:, keep]); gpu=false)
        @test fit_residual(sub, msub, cal, metasub) < 1e-6
    end

    @testset "flagged data are excluded when the model carries the flags" begin
        meta, model, measured, _ = synthetic_problem()
        bad = deepcopy_gpu(measured)
        rng = MersenneTwister(7)
        bad.flags .= rand(rng, size(bad.flags)...) .< 0.3
        bad.xx[bad.flags] .= 1e6
        m = deepcopy_gpu(model)
        m.flags .= bad.flags
        cal = identity_gains(meta)
        solve_gains!(cal, bad, m, meta; maxiter=30, tolerance=1e-10)
        @test fit_residual(bad, m, cal, meta) < 1e-6        # residual over unflagged data only
    end

    @testset "smooth_flux: per-channel gains tied to a band gain" begin
        chans = collect(range(50e6, 52e6; length=6))
        meta, model, _, _ = synthetic_problem(channels=chans)        # ~1000 Jy source
        truth = random_gains(meta.Nant, meta.Nfreq; seed=14)
        measured = deepcopy_gpu(model); gpu_corrupt!(measured, truth, meta)
        spread(c) = maximum(abs, c.xx .- sum(c.xx; dims=2) ./ size(c.xx, 2))
        for diagonal in (false, true), solver in (:rk4, :damped)
            fit(; kw...) = (c = identity_gains(meta; diagonal=diagonal);
                            solve_gains!(c, measured, model, meta; solver=solver,
                                         maxiter=solver == :rk4 ? 60 : 400, tolerance=1e-12, kw...); c)
            r(c) = fit_residual(measured, model, c, meta)
            free = fit()
            off = fit(smooth_flux=0.0)
            @test off.xx == free.xx && off.yy == free.yy                    # 0 = off
            weak = fit(smooth_flux=1e-3)                                    # ρ ≪ D: per channel
            @test abs(r(weak) - r(free)) < 1e-3 * r(free) + 1e-9
            strong = fit(smooth_flux=1e5)                                   # ρ ≫ D: one band gain
            @test spread(strong) < 1e-3 * spread(free)
            mid = fit(smooth_flux=300.0)
            @test r(free) < r(mid) < r(strong) && spread(mid) < spread(free)
        end
        # ρ → ∞ is the one-gain-per-band fit (relative singularity test: no freeze at huge ρ)
        band = identity_gains(meta); solve_gains!(band, measured, model, meta; maxiter=60, tolerance=1e-12,
                                                  smooth_flux=1e9)
        @test spread(band) < 1e-6 && maximum(abs, band.xx .- 1) > 1e-3
    end

    @testset "smooth_flux: flagged channel and antenna without data" begin
        chans = collect(range(50e6, 52e6; length=6))
        meta, model, _, _ = synthetic_problem(channels=chans)
        truth = random_gains(meta.Nant, meta.Nfreq; seed=15)
        measured = deepcopy_gpu(model); gpu_corrupt!(measured, truth, meta)
        measured.flags[:, 3] .= true                                  # channel 3 fully flagged
        bl = meta.baselines
        measured.flags[(bl[1, :] .== 5) .| (bl[2, :] .== 5), :] .= true   # antenna 5 has no data
        m = deepcopy_gpu(model); m.flags .= measured.flags
        for diagonal in (false, true)
            c = identity_gains(meta; diagonal=diagonal)
            solve_gains!(c, measured, m, meta; maxiter=30, tolerance=1e-10, smooth_flux=300.0)
            @test all(isfinite, c.xx) && all(isfinite, c.yy)
            @test all(c.xx[5, :] .== 1) && all(c.yy[5, :] .== 1)      # untouched
            # the flagged channel carries the band gain: between its neighbours' values
            @test abs(c.xx[1, 3] - sum(c.xx[1, [1, 2, 4, 5, 6]]) / 5) < 0.5 * maximum(abs, c.xx[1, :] .- c.xx[1, 3]) + 1e-12
        end
    end

    @testset "precision=:mixed: single-precision normal equations" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        for diagonal in (false, true), smooth in (0.0, 300.0)
            d = identity_gains(meta; diagonal=diagonal); s = identity_gains(meta; diagonal=diagonal)
            solve_gains!(d, measured, model, meta; maxiter=20, tolerance=1e-8, smooth_flux=smooth)
            solve_gains!(s, measured, model, meta; maxiter=20, tolerance=1e-8, smooth_flux=smooth, precision=:mixed)
            @test maximum(abs, s.xx .- d.xx) < 1e-4 * maximum(abs, d.xx)
            @test fit_residual(measured, model, s, meta) < 1.0001 * fit_residual(measured, model, d, meta)
        end
        @test_throws ArgumentError solve_gains!(identity_gains(meta), measured, model, meta; precision=:half)
        # tight tolerances are floored at 1e-6: mixed still converges
        c = identity_gains(meta)
        converged, _, _ = solve_gains!(c, measured, model, meta; maxiter=60, tolerance=1e-10, precision=:mixed)
        @test converged
    end

    @testset "active mask: frozen channels give bitwise the same result" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        g = identity_gains(meta)
        sqm = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false); host_makesquare!(sqm, measured, meta)
        sqo = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false); host_makesquare!(sqo, model, meta)
        k_all = ntuple(_ -> zeros(ComplexF64, meta.Nant, meta.Nfreq), 4); k_act = deepcopy(k_all)
        gg = (g.xx, g.xy, g.yx, g.yy)
        stefcal_steps!(k_all, gg, false, sqm, sqo, meta.Nant)
        active = trues(meta.Nfreq); active[2] = false
        stefcal_steps!(k_act, gg, false, sqm, sqo, meta.Nant; active=active)
        @test all(k_act[1][:, 2] .== 0) && k_act[1][:, active] == k_all[1][:, active]
    end

    @testset "invalid arguments" begin
        meta, model, measured, _ = synthetic_problem()
        @test_throws ArgumentError solve_gains!(identity_gains(meta), measured, model, meta; solver=:newton)
        wide = GPUCalibration(meta.Nant, 1; diagonal=false, gpu=false)
        @test_throws ArgumentError solve_gains!(wide, measured, model, meta)
        prior_1ch = phase_regularization(meta, 1, 1; gpu=false, nfreq=1)       # Real arguments accepted
        @test_throws ArgumentError solve_gains!(identity_gains(meta), measured, model, meta; reg=prior_1ch)
        @test_throws ArgumentError phase_regularization(meta, 1.0, 1.0; gpu=false, nfreq=2)
        sq1 = GPUSquareVisibilities(meta.Nant, 1; gpu=false)
        @test_throws DimensionMismatch host_makesquare!(sq1, measured, meta)
        @test solve_gains!(identity_gains(meta), measured, model, meta; minuvw=5, tolerance=1)[2] >= 1
        @test_throws ArgumentError solve_gains!(identity_gains(meta), measured, model, meta; smooth_flux=-1)
        prior = phase_regularization(meta, 1, 1; gpu=false)
        @test_throws ArgumentError solve_gains!(identity_gains(meta), measured, model, meta;
                                                smooth_flux=100, reg=prior)
    end

    if is_gpu_available()
        @testset "GPU solve matches the host solve" begin
            meta, model, measured, _ = synthetic_problem(noise=2.0)
            gmeta = synthetic_meta(gpu=true)
            up(v) = GPUVisibilities(CuArray(v.xx), CuArray(v.xy), CuArray(v.yx), CuArray(v.yy), CuArray(v.flags))
            for (solver, reg) in ((:rk4, nothing), (:damped, nothing), (:rk4, :prior))
                hreg = reg === nothing ? nothing : phase_regularization(meta, 0.5, 0.5; gpu=false)
                greg = reg === nothing ? nothing : phase_regularization(gmeta, 0.5, 0.5; gpu=true)
                h = identity_gains(meta)
                g = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=true)
                solve_gains!(h, measured, model, meta; solver=solver, maxiter=10, minuvw=2.0, reg=hreg)
                solve_gains!(g, up(measured), up(model), gmeta; solver=solver, maxiter=10, minuvw=2.0, reg=greg)
                for f in (:xx, :xy, :yx, :yy)
                    @test maximum(abs, Array(getfield(g, f)) .- getfield(h, f)) < 1e-9
                end
            end
            hd = identity_gains(meta; diagonal=true)
            gd = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=true, gpu=true)
            solve_gains!(hd, measured, model, meta; maxiter=10)
            solve_gains!(gd, up(measured), up(model), gmeta; maxiter=10)
            @test maximum(abs, Array(gd.xx) .- hd.xx) < 1e-9 && maximum(abs, Array(gd.yy) .- hd.yy) < 1e-9
            for prec in (:double, :mixed)                                  # mixed precision on the GPU
                h = identity_gains(meta); g = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=true)
                solve_gains!(h, measured, model, meta; maxiter=10, minuvw=2.0, precision=prec)
                solve_gains!(g, up(measured), up(model), gmeta; maxiter=10, minuvw=2.0, precision=prec)
                @test maximum(abs, Array(g.xx) .- h.xx) < (prec == :double ? 1e-9 : 1e-4)
            end
            for diagonal in (false, true), solver in (:rk4, :damped)      # consensus step
                h = identity_gains(meta; diagonal=diagonal)
                g = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=diagonal, gpu=true)
                solve_gains!(h, measured, model, meta; solver=solver, maxiter=10, minuvw=2.0, smooth_flux=30.0)
                solve_gains!(g, up(measured), up(model), gmeta; solver=solver, maxiter=10, minuvw=2.0, smooth_flux=30.0)
                for f in (diagonal ? (:xx, :yy) : (:xx, :xy, :yx, :yy))
                    @test maximum(abs, Array(getfield(g, f)) .- getfield(h, f)) < 1e-9
                end
            end
            # The GPU makesquare kernel and the host loop give the same matrices.
            hs = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=false)
            gs = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=true)
            vis = random_vis(meta.Nbase, meta.Nfreq; flag_fraction=0.2)
            host_makesquare!(hs, vis, meta)
            gpu_makesquare!(gs, up(vis), gmeta)
            @test Array(gs.xx) == hs.xx && Array(gs.yx) == hs.yx
            mask_short_baselines!(hs, meta, 30.0)
            mask_short_baselines!(gs, gmeta, 30.0)
            @test Array(gs.xy) == hs.xy
        end
    end
end
