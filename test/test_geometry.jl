@testset "Geometry" begin
    @testset "j2000_to_itrf_matrix is a rotation" begin
        R = j2000_to_itrf_matrix(TEST_MJD)
        @test R' * R ≈ I atol = 1e-12
        @test det(R) ≈ 1 atol = 1e-12
    end

    @testset "precession of the pole since J2000" begin
        # Mostly precession: ~20″/yr × 25 yr ≈ 0.14°, plus ≲ 10″ nutation.
        R = j2000_to_itrf_matrix(TEST_MJD)
        angle = rad2deg(acos(clamp(dot(R * [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]), -1, 1)))
        @test 0.12 < angle < 0.16
    end

    @testset "Earth rotation: one sidereal day" begin
        R1 = j2000_to_itrf_matrix(TEST_MJD)
        R2 = j2000_to_itrf_matrix(TEST_MJD + 0.99726958)
        @test maximum(abs, R1 .- R2) < 1e-5
        R3 = j2000_to_itrf_matrix(TEST_MJD + 0.5 * 0.99726958)
        @test maximum(abs, R1 .- R3) > 1     # half a turn about z
    end

    @testset "uvw_geometry recovers the phase centre and the MS sign" begin
        meta = synthetic_meta()
        s0, sign, rms = uvw_geometry(meta)
        ra, dec = meta.phase_center_ra, meta.phase_center_dec
        s0_true = j2000_to_itrf_matrix(TEST_MJD) * [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)]
        @test sign == -1
        @test norm(s0 - s0_true) < 1e-9
        @test rms < 1e-6
    end

    @testset "geometric_uvw!" begin
        meta = synthetic_meta()
        uvw_ms = copy(meta.uvw)
        rms_w, offset = geometric_uvw!(meta, TEST_MJD)
        @test rms_w < 1e-6           # same w as the MS, opposite sign convention
        @test offset < 1e-6
        @test meta.uvw[3, :] ≈ -uvw_ms[3, :] atol = 1e-6
        pos, bl = meta.antenna_positions, meta.baselines
        for α in (1, 17, size(bl, 2))
            b = pos[:, bl[2, α]] .- pos[:, bl[1, α]]
            @test norm(meta.uvw[:, α]) ≈ norm(b) rtol = 1e-12
        end
    end
end
