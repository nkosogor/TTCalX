# zest_sky_rfi!(...; peel_fast=true): off by default (nothing changes), the argument checks, and on the
# GPU the same peel as the default to Float32 rounding: the same guard decisions, convergence flags and
# iteration counts, every row peeled (also the fully flagged ones), ComplexF32 visibilities in and out.
# Uses ws_problem / ws_run / same (test_workspace.jl) and synthetic_problem (test_solver.jl).

const FAST_CFG = (precision=:mixed, smooth_flux=0.0, solver=:rk4)

"""ws_problem with one antenna flagged everywhere and two baselines flagged in every channel."""
function fast_problem(; seed=1, gpu=true)
    meta, measured, sky, rfi, _ = ws_problem(; seed=seed, gpu=gpu)
    bl = Array(meta.baselines)
    fl = Array(measured.flags)
    fl[(bl[1, :] .== 3) .| (bl[2, :] .== 3), :] .= true          # a dead antenna
    fl[[11, 40], :] .= true                                         # two dead baselines
    copyto!(measured.flags, fl)
    meta, measured, sky, rfi
end

to32(v::GPUVisibilities) = GPUVisibilities(ComplexF32.(v.xx), ComplexF32.(v.xy), ComplexF32.(v.yx), ComplexF32.(v.yy),
                                           copy(v.flags))

"""Max |a - b| over the four correlations, relative to max |b|, on the rows `rows`."""
relrows(a, b, rows) = maximum(maximum(abs, ComplexF64.(getfield(a, f)[rows, :]) .- getfield(b, f)[rows, :])
                              for f in (:xx, :xy, :yx, :yy)) /
                      maximum(maximum(abs, getfield(b, f)[rows, :]) for f in (:xx, :xy, :yx, :yy))

@testset "peel_fast" begin
    @test !TTCalX.PEEL_FAST[]
    @testset "argument checks (host)" begin
        meta, measured, sky, rfi = fast_problem(; gpu=false)
        @test_throws ArgumentError zest_sky_rfi!(deepcopy_gpu(measured), meta, TEST_MJD, sky, rfi;
                                                 FAST_CFG..., workspace=PeelWorkspace(), peel_fast=true)
        @test !TTCalX.PEEL_FAST[] && !TTCalX._PEEL_FAST_BUSY[]
    end
    is_gpu_available() || return
    try
        fast_kernels!(true)
        meta, measured, sky, rfi = fast_problem()
        ws = PeelWorkspace()
        peelrun(v; kw...) = ws_run(meta, v, sky, rfi, GPUSource[]; FAST_CFG..., workspace=ws, kw...)
        ref = peelrun(measured)                              # the default peel
        @testset "argument checks (GPU)" begin
            for kw in ((workspace=nothing,), (solver=:damped,), (precision=:double,), (diagonal=true,),
                       (smooth_flux=100.0,), (maxiter=TTCalX.FAST_MAXITER + 1,),
                       (reg=(0.5, 0.5),))
                @test_throws ArgumentError peelrun(measured; peel_fast=true, kw...)
            end
            @test_throws ArgumentError ws_run(meta, measured, sky, rfi, [sky[1]]; FAST_CFG..., workspace=ws,
                                              peel_fast=true)
            fast_kernels!(false)
            @test_throws ArgumentError peelrun(measured; peel_fast=true)
            fast_kernels!(true)
            # one fast peel per process at a time
            TTCalX._PEEL_FAST_BUSY[] = true
            @test_throws ErrorException peelrun(measured; peel_fast=true)
            TTCalX._PEEL_FAST_BUSY[] = false
            @test !TTCalX.PEEL_FAST[]
        end
        alive = vec(any(.!Array(measured.flags); dims=2))
        live, dead = findall(alive), findall(.!alive)
        @test !isempty(dead)
        for (name, input) in (("ComplexF64", measured), ("ComplexF32", to32(measured)))
            @testset "agrees with the default peel, $name visibilities" begin
                vis = deepcopy_gpu(input)
                log = []
                r = zest_sky_rfi!(vis, meta, TEST_MJD, sky, rfi; FAST_CFG..., peeliter=2, maxiter=5, workspace=ws,
                                  peel_fast=true, report=(n, it; kw...) -> push!(log, (n, it, sort(collect(kw); by=first))))
                @test !TTCalX.PEEL_FAST[] && !TTCalX._PEEL_FAST_BUSY[]
                @test eltype(vis.xx) == eltype(input.xx)
                @test r.peeled == ref.peeled && r.skipped == ref.skipped && r.flagged_zeros == ref.flagged
                @test Array(vis.flags) == ref.vis.flags
                # same decisions, solves and convergence flags
                @test [(l[1], l[2], Dict(l[3])[:action], Dict(l[3])[:converged], Dict(l[3])[:niters]) for l in log] ==
                      [(l[1], l[2], Dict(l[3])[:action], Dict(l[3])[:converged], Dict(l[3])[:niters]) for l in ref.log]
                h = host_copy(vis)
                @test relrows(h, ref.vis, live) < 2e-5
                @test relrows(h, ref.vis, dead) < 2e-5        # the fully flagged rows are peeled too
                for (n, c) in r.gains.cals
                    @test maximum(abs, Array(c.xx) .- ref.gains[n].xx) < 1e-4
                end
                # deterministic: a second run is bitwise the first
                v2 = deepcopy_gpu(input)
                zest_sky_rfi!(v2, meta, TEST_MJD, sky, rfi; FAST_CFG..., peeliter=2, maxiter=5, workspace=ws,
                              peel_fast=true)
                @test same(host_copy(v2), h)
            end
        end
        @testset "the default after a fast peel is unchanged (bitwise)" begin
            @test same(peelrun(measured), ref)
        end
        @testset "fully flagged snapshot" begin
            v = deepcopy_gpu(measured); v.flags .= true
            d = deepcopy_gpu(v)
            a = zest_sky_rfi!(v, meta, TEST_MJD, sky, rfi; FAST_CFG..., workspace=ws, peel_fast=true)
            b = zest_sky_rfi!(d, meta, TEST_MJD, sky, rfi; FAST_CFG..., workspace=ws)
            @test isempty(a.peeled) && a.peeled == b.peeled
            @test relrows(host_copy(v), host_copy(d), 1:meta.Nbase) < 1e-5
        end
        @testset "RK4 solve: converged flag and iterations as solve_gains!" begin
            pmeta, model, pmeas, _ = synthetic_problem(; noise=0.1)
            gmeta = synthetic_meta(; gpu=true)
            up(v) = GPUVisibilities(CuArray(v.xx), CuArray(v.xy), CuArray(v.yx), CuArray(v.yy), CuArray(v.flags))
            gmod, gmeas = up(model), up(pmeas)
            for (maxiter, tol) in ((30, 1e-4), (2, 1e-4), (5, 1e-12), (TTCalX.FAST_MAXITER, 1e-5))
                c0 = GPUCalibration(gmeta.Nant, gmeta.Nfreq; diagonal=false, gpu=true)
                c1 = GPUCalibration(gmeta.Nant, gmeta.Nfreq; diagonal=false, gpu=true)
                r0 = solve_gains!(c0, gmeas, gmod, gmeta; solver=:rk4, maxiter=maxiter, tolerance=tol,
                                  minuvw=5.0, precision=:mixed, workspace=ws)
                w2 = PeelWorkspace()
                msq = TTCalX._ws_square!(w2, :meas_sq, gmeta.Nant, gmeta.Nfreq, ComplexF32, true; zero=false)
                r1 = try
                    TTCalX.PEEL_FAST[] = true
                    TTCalX._fast_solve!(c1, gmeas, gmod, gmeta, nothing, msq, w2, nothing; maxiter=maxiter,
                                        tolerance=tol, minuvw=5.0)
                finally
                    TTCalX.PEEL_FAST[] = false
                    TTCalX._fast_reset_caches!()
                end
                @test r1[1] == r0[1] && r1[2] == r0[2] && r1[3] ≈ r0[3]
                @test maximum(abs, Array(c1.xx) .- Array(c0.xx)) < 1e-5
            end
            c = GPUCalibration(gmeta.Nant, gmeta.Nfreq; diagonal=false, gpu=true)
            @test_throws ArgumentError TTCalX._fast_solve!(c, gmeas, gmod, gmeta, nothing,
                TTCalX._ws_square!(ws, :meas_sq, gmeta.Nant, gmeta.Nfreq, ComplexF32, true), ws, nothing;
                maxiter=TTCalX.FAST_MAXITER + 1, tolerance=1e-4, minuvw=5.0)
        end
    finally
        fast_kernels!(false)
    end
end
