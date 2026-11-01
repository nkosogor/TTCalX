# PeelWorkspace: zest_sky_rfi! / solve_gains! with reused buffers give bitwise the same
# results as without, over a sequence of calls with the same shapes and after shape changes.
# Uses synthetic_problem (test_solver.jl) and ground_rfi (test_sky_rfi.jl).

"""Two sky sources + an RFI emitter + Jupiter-like restore source, data varied by `seed`."""
function ws_problem(; seed=1, Nant=24, channels=[40e6, 55e6, 70e6], gpu=false)
    meta, _, measured, _ = synthetic_problem(; seed=seed, Nant=Nant, channels=channels, noise=0.5)
    a = GPUPointSource("a", azel_radec(80.0, 60.0)..., GPUPowerLaw(1000.0, 20.0, 10.0, 5.0, 50e6, [-0.7]))
    b = GPUPointSource("b", azel_radec(300.0, 35.0)..., GPUPowerLaw(400.0, 0.0, 0.0, 0.0, 50e6, [-0.8]))
    low = GPUPointSource("low", azel_radec(200.0, -5.0)..., GPUPowerLaw(500.0, 0.0, 0.0, 0.0, 50e6, [-0.7]))
    jup = GPUPointSource("jup", azel_radec(250.0, 40.0)..., GPUPowerLaw(300.0, 0.0, 0.0, 120.0, 50e6, [0.0]))
    for (s, g) in ((b, 11), (jup, 12))
        m = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(m, meta, s, TEST_MJD)
        gpu_corrupt!(m, random_gains(meta.Nant, meta.Nfreq; seed=seed + g, ε=0.05), meta)
        gpu_putsrc!(measured, m)
    end
    rfi = ground_rfi()
    m = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
    gpu_genvis!(m, meta, rfi, meta.phase_center_ra, meta.phase_center_dec, 0.0)
    gpu_corrupt!(m, random_gains(meta.Nant, meta.Nfreq; seed=seed + 13, ε=0.05), meta)
    gpu_putsrc!(measured, m)
    measured.xx[2 + seed % 5, :] .= 0; measured.yy[2 + seed % 5, :] .= 0      # dropped data
    measured.flags[7 + seed % 3, 1] = true
    if gpu
        meta = synthetic_meta(; Nant=Nant, channels=channels, gpu=true)
        measured = GPUVisibilities(CuArray(measured.xx), CuArray(measured.xy), CuArray(measured.yx),
                                   CuArray(measured.yy), CuArray(measured.flags))
    end
    meta, measured, [a, b, low], [rfi], [jup]
end

"""Everything a zest returns or writes, on the host: visibilities, gains, report log."""
function ws_run(meta, measured, sky, rfi, restore; workspace=nothing, kw...)
    vis = deepcopy_gpu(measured)
    log = []
    r = zest_sky_rfi!(vis, meta, TEST_MJD, sky, rfi; restore=restore, restore_offset_deg=0.3,
                      peeliter=2, maxiter=5, workspace=workspace,
                      report=(name, it; kw...) -> push!(log, (name, it, sort(collect(kw); by=first))), kw...)
    gains = Dict(n => host_copy(c) for (n, c) in r.gains.cals)
    (vis=host_copy(vis), log=log, peeled=r.peeled, skipped=r.skipped, flagged=r.flagged_zeros,
     restored=r.restored, gains=gains)
end

"""Bitwise equality (=== on the floats, so NaN and -0.0 count)."""
same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(isequal.(a, b))
same(a::GPUVisibilities, b::GPUVisibilities) =
    all(same(getfield(a, f), getfield(b, f)) for f in (:xx, :xy, :yx, :yy, :flags))
same(a::GPUCalibration, b::GPUCalibration) =
    a.is_diagonal == b.is_diagonal && all(same(getfield(a, f), getfield(b, f)) for f in (:xx, :xy, :yx, :yy, :flags))
function same(a::NamedTuple, b::NamedTuple)
    same(a.vis, b.vis) && isequal(a.log, b.log) && a.peeled == b.peeled && a.skipped == b.skipped &&
        a.flagged == b.flagged && isequal(a.restored, b.restored) && keys(a.gains) == keys(b.gains) &&
        all(same(a.gains[k], b.gains[k]) for k in keys(a.gains))
end

const WS_CONFIGS = (
    (precision=:mixed, smooth_flux=300.0),                      # the pipeline's settings
    (precision=:double, smooth_flux=0.0),
    (precision=:mixed, smooth_flux=0.0, cache_model_sq=false, rfi_mode=:joint),
    (precision=:double, diagonal=true, solver=:damped),
)

function ws_sequence_tests(; gpu=false)
    for cfg in WS_CONFIGS
        ws = PeelWorkspace()
        nalloc = Int[]
        for seed in 1:4                                  # same shapes: buffers reused
            p = ws_problem(; seed=seed, gpu=gpu)
            ref = ws_run(p...; cfg...)
            got = ws_run(p...; workspace=ws, cfg...)
            @test same(got, ref)
            push!(nalloc, ws.nalloc)
        end
        @test nalloc[1] > 0 && all(==(nalloc[1]), nalloc)  # nothing new after the first call
        @test ws.nreset == 0 && workspace_bytes(ws) > 0
        # other shapes: the workspace starts again, and returns to the first shape
        for (k, shape) in enumerate(((Nant=20, channels=[45e6, 60e6]), (Nant=24, channels=[40e6, 55e6, 70e6])))
            p = ws_problem(; seed=10 + k, gpu=gpu, shape...)
            @test same(ws_run(p...; workspace=ws, cfg...), ws_run(p...; cfg...))
            @test ws.nreset == k
        end
        # fewer directions (low source and RFI dropped): uses a subset of the buffers
        meta, measured, sky, _, restore = ws_problem(; seed=20, gpu=gpu)
        n0 = ws.nalloc
        @test same(ws_run(meta, measured, sky[1:1], GPUSource[], restore; workspace=ws, cfg...),
                   ws_run(meta, measured, sky[1:1], GPUSource[], restore; cfg...))
        @test ws.nalloc == n0
        free_workspace!(ws)
        @test workspace_bytes(ws) == 0 && ws.key === nothing
    end
end

@testset "PeelWorkspace" begin
    @testset "zest_sky_rfi! sequence, host" begin
        ws_sequence_tests(gpu=false)
    end

    @testset "solve_gains! with a workspace" begin
        meta, model, measured, _ = synthetic_problem(noise=0.3)
        ws = PeelWorkspace()
        for kw in ((solver=:rk4,), (solver=:damped,), (solver=:rk4, smooth_flux=100.0),
                   (solver=:rk4, precision=:mixed), (solver=:rk4, reg=phase_regularization(meta, 0.5, 0.5; gpu=false)))
            for diagonal in (false, true)
                cal() = GPUCalibration(meta.Nant, meta.Nfreq; diagonal=diagonal, gpu=false)
                a, b, c = cal(), cal(), cal()
                ra = solve_gains!(a, measured, model, meta; minuvw=5.0, kw...)
                rb = solve_gains!(b, measured, model, meta; minuvw=5.0, workspace=ws, kw...)
                rc = solve_gains!(c, measured, model, meta; minuvw=5.0, workspace=ws, kw...)   # reused
                @test ra == rb == rc && same(a, b) && same(a, c)
            end
        end
    end

    @testset "residual_power! and flag_zero_visibilities! with buffers" begin
        vis = random_vis(50, 3; flag_fraction=0.2)
        vis.xx[4, :] .= 0; vis.yy[4, :] .= 0
        @test TTCalX.residual_power!(similar(vis.xx), vis) === residual_power(vis)
        a, b = deepcopy_gpu(vis), deepcopy_gpu(vis)
        @test flag_zero_visibilities!(a) == flag_zero_visibilities!(b, PeelWorkspace())
        @test same(a, b)
    end

    if is_gpu_available()
        @testset "zest_sky_rfi! sequence, GPU" begin
            ws_sequence_tests(gpu=true)
        end
        @testset "warm GPU zest allocates little device memory" begin
            p = ws_problem(; seed=3, gpu=true)
            ws = PeelWorkspace()
            ws_run(p...; workspace=ws, precision=:mixed, smooth_flux=300.0)        # compile, fill ws
            ws_run(p...; precision=:mixed, smooth_flux=300.0)
            run(vis, w) = zest_sky_rfi!(vis, p[1], TEST_MJD, p[3], p[4]; restore=p[5], restore_offset_deg=0.3,
                                        peeliter=2, maxiter=5, precision=:mixed, smooth_flux=300.0, workspace=w)
            a, b = deepcopy_gpu(p[2]), deepcopy_gpu(p[2])
            cold = CUDA.@allocated run(a, nothing)
            warm = CUDA.@allocated run(b, ws)
            @test same(host_copy(a), host_copy(b))
            @test warm < 0.2 * cold
        end
    end
end
