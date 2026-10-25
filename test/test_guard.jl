# Divergence guards: zero flagging and the residual-power guard of guarded_zest!.
# Uses synthetic_problem from test_solver.jl (included before this file).

"""A solver that 'diverges': scales the gains of a few antennas by `factor`."""
diverging_solve(factor) = (cal, vis, coh) -> begin
    cal.xx[1:3, :] .*= factor; cal.yy[1:3, :] .*= factor; cal.xy[1:3, :] .+= factor
    false, 5
end

@testset "Guards" begin
    @testset "flag_zero_visibilities!" begin
        meta = synthetic_meta()
        vis = random_vis(meta.Nbase, meta.Nfreq; flag_fraction=0.1)
        vis.xx[5, :] .= 0; vis.yy[5, :] .= 0           # dropped baseline
        vis.xx[7, 2] = 0                               # XX only: kept
        already = count(vis.flags[5, :])
        f72 = vis.flags[7, 2]
        n = flag_zero_visibilities!(vis)
        @test n == meta.Nfreq - already
        @test all(vis.flags[5, :])
        @test vis.flags[7, 2] == f72
        @test flag_zero_visibilities!(vis) == 0
        @test residual_power(vis) ≈ sum(abs2, vis.xx .* .!vis.flags) + sum(abs2, vis.yy .* .!vis.flags)
    end

    @testset "guarded_zest! accepts a good solve" begin
        meta, model, measured, truth = synthetic_problem()
        vis = deepcopy_gpu(measured)
        cal = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=false)
        solve = (c, v, m) -> begin
            r = solve_gains!(c, v, m, meta; solver=:rk4, maxiter=30, tolerance=1e-8)
            r[1], r[2]
        end
        actions = Symbol[]
        sub = guarded_zest!(vis, meta, [model], [cal], solve; peeliter=2,
                            report=(s, it; action, kw...) -> push!(actions, action))
        @test sub == [true]
        @test actions[1] == :accept
        @test residual_power(vis) < 1e-6 * residual_power(measured)
    end

    @testset "guarded_zest! rejects a diverging solve" begin
        meta, model, measured, truth = synthetic_problem()
        # Data: one source; the model also has a second direction that is not in
        # the data and whose solver diverges.  Unguarded, peeling it raises the power.
        ra2, dec2 = azel_radec(300.0, 2.0)               # a "setting" source not in the data
        src2 = GPUPointSource("src2", ra2, dec2, GPUPowerLaw(500.0, 0.0, 0.0, 0.0, 50e6, [-0.7]))
        model2 = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(model2, meta, src2, TEST_MJD)
        vis = deepcopy_gpu(measured)
        p0 = residual_power(vis)
        cals = [GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=false) for _ in 1:2]
        good = (c, v, m) -> begin
            r = solve_gains!(c, v, m, meta; solver=:rk4, maxiter=30, tolerance=1e-8)
            r[1], r[2]
        end
        bad = diverging_solve(30.0)
        calls = Ref(0)
        solve = (c, v, m) -> (calls[] += 1; m === model ? good(c, v, m) : bad(c, v, m))
        log = []
        sub = guarded_zest!(vis, meta, [model, model2], cals, solve; peeliter=3,
                            report=(s, it; action, p_with, p_new, kw...) -> push!(log, (s, it, action, p_with, p_new)))
        @test calls[] == 6
        @test sub[1] && !sub[2]                         # diverged source left in
        @test all(e[3] != :accept for e in log if e[1] == 2)
        @test all(abs.(Array(cals[2].xx)) .≈ 1)         # its gains restored
        # The residual never exceeds the power with every source left in ...
        @test residual_power(vis) < p0
        # ... and the modelled source is still peeled to the noise floor.
        @test residual_power(vis) < 1e-12 * p0

        # Unguarded reference (CPU TTCal's loop): the same solves make it worse.
        vis_u = deepcopy_gpu(measured)
        cals_u = [GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=false) for _ in 1:2]
        for (m, c) in zip([model, model2], cals_u); TTCalX._subtract!(vis_u, m, c, meta); end
        for _ in 1:3, (m, c) in zip([model, model2], cals_u)
            TTCalX._add!(vis_u, m, c, meta); solve(c, vis_u, m); TTCalX._subtract!(vis_u, m, c, meta)
        end
        @test residual_power(vis_u) > 100 * residual_power(vis)
    end

    @testset "guarded_zest! max_gain and require_convergence" begin
        meta, model, measured, _ = synthetic_problem()
        vis = deepcopy_gpu(measured)
        cal = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=false)
        acts = Symbol[]
        guarded_zest!(vis, meta, [model], [cal], diverging_solve(1.01); peeliter=1, max_gain=1.0,
                      report=(s, it; action, kw...) -> push!(acts, action))
        @test acts == [:keep_previous] || acts == [:leave_in]
        @test all(Array(cal.xx) .== 1)
        acts = Symbol[]
        guarded_zest!(deepcopy_gpu(measured), meta, [model],
                      [GPUCalibration(meta.Nant, meta.Nfreq; diagonal=false, gpu=false)],
                      (c, v, m) -> (false, 5); peeliter=1, require_convergence=true,
                      report=(s, it; action, kw...) -> push!(acts, action))
        @test acts[1] != :accept
    end
end
