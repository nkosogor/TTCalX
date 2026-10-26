@testset "Host backend" begin
    meta = synthetic_meta()
    @testset "threaded gpu_corrupt! equals cpu_corrupt!" begin
        for diagonal in (false, true)
            vis = random_vis(meta.Nbase, meta.Nfreq)
            cal = random_gains(meta.Nant, meta.Nfreq)
            cal = GPUCalibration(cal.xx, diagonal ? zero(cal.xy) : cal.xy, diagonal ? zero(cal.yx) : cal.yx,
                                 cal.yy, cal.flags, diagonal)
            a = deepcopy_gpu(vis); b = deepcopy_gpu(vis)
            gpu_corrupt!(a, cal, meta)
            TTCalX.cpu_corrupt!(b, cal, meta)
            @test a.xx == b.xx && a.xy == b.xy && a.yx == b.yx && a.yy == b.yy
        end
    end

    @testset "subtract and add back" begin
        data = random_vis(meta.Nbase, meta.Nfreq; seed=11)
        model = random_vis(meta.Nbase, meta.Nfreq; seed=12)
        orig = host_copy(data)
        gpu_subsrc!(data, model)
        @test data.xx ≈ orig.xx .- model.xx
        gpu_putsrc!(data, model)
        @test data.xx ≈ orig.xx && data.yy ≈ orig.yy
    end

    @testset "peel one source on the host" begin
        # Subtract a source corrupted by unknown gains, as one zest pass does.
        meta2, model, measured, _ = synthetic_problem(noise=0.0)
        cal = GPUCalibration(meta2.Nant, meta2.Nfreq; diagonal=false, gpu=false)
        solve_gains!(cal, measured, model, meta2; maxiter=30, tolerance=1e-10)
        corrupted = deepcopy_gpu(model)
        gpu_corrupt!(corrupted, cal, meta2)
        residual = deepcopy_gpu(measured)
        gpu_subsrc!(residual, corrupted)
        @test norm(residual.xx) < 1e-8 * norm(measured.xx)
    end
end
