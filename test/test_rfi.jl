@testset "RFI sources" begin
    @testset "wgs84_to_itrf" begin
        @test wgs84_to_itrf(0.0, 0.0, 0.0) ≈ [6378137.0, 0.0, 0.0]
        @test wgs84_to_itrf(0.0, 90.0, 0.0) ≈ [0.0, 0.0, 6356752.314245] atol = 1e-3
        @test wgs84_to_itrf(90.0, 0.0, 10.0) ≈ [0.0, 6378147.0, 0.0] atol = 1e-6
        # height moves the point along the ellipsoid normal
        p0 = wgs84_to_itrf(-118.28, 37.24, 0.0)
        p1 = wgs84_to_itrf(-118.28, 37.24, 100.0)
        @test norm(p1 - p0) ≈ 100 rtol = 1e-9        # ~6.4e6 m coordinates
    end

    @testset "spectrum: clamp and linear interpolation (ttcal_dev's RFISpectrum)" begin
        sp = GPURFISpectrum([10e6, 20e6, 30e6], [1.0, 3.0, 5.0], zeros(3), zeros(3), [0.5, 1.5, 2.5])
        @test sp(5e6) == (1.0, 0.0, 0.0, 0.5)
        @test sp(35e6) == (5.0, 0.0, 0.0, 2.5)
        @test sp(20e6) == (3.0, 0.0, 0.0, 1.5)
        @test all(sp(15e6) .≈ (2.0, 0.0, 0.0, 1.0))
    end

    @testset "construct from TTCal JSON" begin
        c = Dict("name" => "test", "sys" => "WGS84", "long" => -118.6, "lat" => 37.5, "el" => 2000.0,
                 "rfi-frequencies" => [10e6, 20e6], "rfi-I" => [1.0, 2.0], "rfi-V" => [0.1, 0.2])
        src = TTCalX.construct_gpu_rfi_source(c)
        @test src isa GPURFISource
        @test is_rfi(src) && !is_rfi(GPUPointSource("p", 0.0, 0.0, GPUPowerLaw(1, 0, 0, 0, 1e6, [0.0])))
        @test src.position_itrf ≈ wgs84_to_itrf(-118.6, 37.5, 2000.0)
        @test src.spectrum(10e6) == (1.0, 0.0, 0.0, 0.1)
        itrf = TTCalX.construct_gpu_rfi_source(merge(c, Dict("sys" => "ITRF", "el" => 1.0, "long" => 2.0, "lat" => 3.0)))
        @test itrf.position_itrf == [1.0, 2.0, 3.0]
        @test_throws ErrorException TTCalX.construct_gpu_rfi_source(merge(c, Dict("sys" => "B1950")))
        @test_throws ArgumentError TTCalX.construct_gpu_rfi_source(merge(c, Dict("rfi-V" => [0.1])))
        meta = synthetic_meta()
        @test_throws ArgumentError genvis_exact!(GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false), meta, src, TEST_MJD)

        path = tempname() * ".json"
        write(path, """[{"name": "rfi", "sys": "WGS84", "long": -118.6, "lat": 37.5, "el": 2000.0,
                         "rfi-frequencies": [1e7, 2e7], "rfi-I": [1.0, 2.0]}]""")
        srcs = read_gpu_sources(path)
        @test length(srcs) == 1 && srcs[1] isa GPURFISource
    end

    @testset "near-field model tends to a plane wave far away" begin
        meta = synthetic_meta()
        geometric_uvw!(meta, TEST_MJD)
        east, north, up = enu_basis()
        d = normalize(cos(deg2rad(30)) * east + sin(deg2rad(30)) * up)    # 30° elevation, due east
        D = 1e9
        src = GPURFISource("far", SVector{3}(Vector(OVRO_ITRF) .+ D .* d),
                           GPURFISpectrum([1e6], [2.0], [0.0], [0.3], [0.5]))
        vis = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        gpu_genvis!(vis, meta, src, 0.0, 0.0, 0.0)
        s0, _, _ = uvw_geometry(meta)
        explicit = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        gpu_genvis!(explicit, meta, src, 0.0, 0.0, 0.0; s0=s0)
        @test explicit.xx == vis.xx
        pos, bl = meta.antenna_positions, meta.baselines
        for β in 1:meta.Nfreq, α in (1, 50, meta.Nbase)
            b = pos[:, bl[2, α]] .- pos[:, bl[1, α]]
            fringe = cis(2π * meta.channels[β] / SPEED_OF_LIGHT * dot(s0 - d, b))
            @test vis.xx[α, β] ≈ 2.0 * fringe rtol = 1e-2
            @test vis.xy[α, β] ≈ complex(0.3, -0.5) * fringe rtol = 1e-2     # xy = U - iV
            @test vis.yx[α, β] ≈ complex(0.3, 0.5) * fringe rtol = 1e-2
        end
    end
end
