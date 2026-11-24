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
                       (reg=(0.5, 0.5),), (rfi_mode=:joint,), (cache_model_sq=false,),
                       (initial_gains=(channels=Float64[], mjd=0.0, cals=Dict()),))
                @test_throws ArgumentError peelrun(measured; peel_fast=true, kw...)
            end
            @test_throws ArgumentError ws_run(meta, measured, sky, rfi, [sky[1]]; FAST_CFG..., workspace=ws,
                                              peel_fast=true)
            @test_throws ArgumentError peelrun(to32(measured))          # ComplexF32 needs peel_fast
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
        @testset "an exception inside resets the state" begin
            fast(v; kw...) = zest_sky_rfi!(v, meta, TEST_MJD, sky, rfi; FAST_CFG..., peeliter=2, workspace=ws,
                                           peel_fast=true, kw...)
            a = deepcopy_gpu(measured); fast(a)
            n = Ref(0)
            boom = (name, it; kw...) -> (n[] += 1; n[] == 3 && error("boom"); nothing)
            @test_throws ErrorException fast(deepcopy_gpu(measured); report=boom)
            @test !TTCalX.PEEL_FAST[] && !TTCalX._PEEL_FAST_BUSY[] && TTCalX._SQ_MASK[] === nothing
            b = deepcopy_gpu(measured); fast(b)
            @test same(host_copy(a), host_copy(b))
            @test same(peelrun(measured), ref)                          # and the default still runs
        end
        @testset "1-2 fully flagged baselines: their rows are peeled as by the default" begin
            for ndead in (1, 2)
                m2, v2, sky2, rfi2, _ = ws_problem(; seed=2, gpu=true)    # one dropped (all-zero) baseline
                ndead == 2 && (v2.flags[30, :] .= true)
                w2 = PeelWorkspace()
                d = ws_run(m2, v2, sky2, rfi2, GPUSource[]; FAST_CFG..., workspace=w2)
                f = deepcopy_gpu(v2)
                r = zest_sky_rfi!(f, m2, TEST_MJD, sky2, rfi2; FAST_CFG..., peeliter=2, maxiter=5, workspace=w2,
                                  peel_fast=true)
                @test get_name(rfi2[1]) in r.peeled && r.peeled == d.peeled
                rows = findall(vec(all(d.vis.flags; dims=2)))
                @test length(rows) == ndead
                @test relrows(host_copy(f), d.vis, rows) < 2e-5
            end
        end
        @testset "workspace reset at the 1024-row padding boundary" begin
            m3, v3, sky3, rfi3, _ = ws_problem(; seed=4, Nant=48, gpu=true)      # 1128 baselines
            v3b = deepcopy_gpu(v3); v3b.flags[1:200, :] .= true                 # < 1024 live rows
            run3(v, w) = (x = deepcopy_gpu(v);
                          zest_sky_rfi!(x, m3, TEST_MJD, sky3, rfi3; FAST_CFG..., peeliter=2, workspace=w, peel_fast=true);
                          host_copy(x))
            w3 = PeelWorkspace()
            a1 = run3(v3, w3); k1 = w3.key; r1 = w3.nreset
            b1 = run3(v3b, w3)
            @test w3.key != k1 && w3.key[2] == 1024 && k1[2] == 2048 && w3.nreset == r1 + 1
            a2 = run3(v3, w3)
            @test same(a1, a2) && same(b1, run3(v3b, PeelWorkspace()))
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
                    TTCalX._fast_solve!(c1, gmeas, gmod, gmeta,
                                        TTCalX._fast_square!(TTCalX._ws_square!(w2, :mod_sq, gmeta.Nant, gmeta.Nfreq,
                                            ComplexF32, true; zero=false), gmod, gmeta, 5.0, nothing),
                                        msq, w2, nothing; maxiter=maxiter,
                                        tolerance=tol, minuvw=5.0)
                finally
                    TTCalX.PEEL_FAST[] = false
                    TTCalX._fast_reset_caches!()
                end
                @test r1[1] == r0[1] && r1[2] == r0[2] && r1[3] ≈ r0[3]
                @test maximum(abs, Array(c1.xx) .- Array(c0.xx)) < 1e-5
            end
            # more than 1024 antennas: the per-channel final kernel with a strided antenna loop
            bmeta, bmodel, bmeas, _ = synthetic_problem(; noise=0.1, Nant=1030, channels=[50e6, 60e6])
            gb = synthetic_meta(; Nant=1030, channels=[50e6, 60e6], gpu=true)
            bm, bv = up(bmodel), up(bmeas)
            c0 = GPUCalibration(gb.Nant, gb.Nfreq; diagonal=false, gpu=true)
            c1 = GPUCalibration(gb.Nant, gb.Nfreq; diagonal=false, gpu=true)
            r0 = solve_gains!(c0, bv, bm, gb; solver=:rk4, maxiter=5, tolerance=1e-4, minuvw=5.0, precision=:mixed,
                              workspace=PeelWorkspace())
            w4 = PeelWorkspace()
            sq(t) = TTCalX._ws_square!(w4, t, gb.Nant, gb.Nfreq, ComplexF32, true; zero=false)
            r1 = try
                TTCalX.PEEL_FAST[] = true
                TTCalX._fast_solve!(c1, bv, bm, gb, TTCalX._fast_square!(sq(:m), bm, gb, 5.0, nothing), sq(:v), w4,
                                    nothing; maxiter=5, tolerance=1e-4, minuvw=5.0)
            finally
                TTCalX.PEEL_FAST[] = false
                TTCalX._fast_reset_caches!()
            end
            @test r1[1] == r0[1] && r1[2] == r0[2] && r1[3] ≈ r0[3]
            @test maximum(abs, Array(c1.xx) .- Array(c0.xx)) < 1e-5
            bm = bv = nothing; GC.gc(); CUDA.reclaim()
            c = GPUCalibration(gmeta.Nant, gmeta.Nfreq; diagonal=false, gpu=true)
            msq = TTCalX._ws_square!(ws, :meas_sq, gmeta.Nant, gmeta.Nfreq, ComplexF32, true)
            @test_throws ArgumentError TTCalX._fast_solve!(c, gmeas, gmod, gmeta, nothing, msq, ws, nothing;
                                                           maxiter=5, tolerance=1e-4, minuvw=5.0)   # no model_sq
            @test_throws "maxiter" TTCalX._fast_solve!(c, gmeas, gmod, gmeta, msq, msq, ws, nothing;
                                                       maxiter=TTCalX.FAST_MAXITER + 1, tolerance=1e-4, minuvw=5.0)
            TTCalX._fast_reset_caches!()
        end
    finally
        fast_kernels!(false)
    end
end
