"""
Straightforward evaluation of CPU TTCal's model (instrument/genvis.jl:
per-antenna delays τ_i = r_i·(ŝ - ŝ₀), fringe e^{2πiντ_i} conj(e^{2πiντ_j})),
used as the reference for `genvis_exact!`.
"""
function reference_exact_model(meta, source, mjd)
    R = j2000_to_itrf_matrix(mjd)
    pos, bl, ch = Array(meta.antenna_positions), Array(meta.baselines), Array(meta.channels)
    unit(ra, dec) = [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)]
    s0 = R * unit(meta.phase_center_ra, meta.phase_center_dec)
    up = normalize(vec(sum(pos, dims=2)))
    out = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
    comps = source isa GPUMultiSource ? source.components : [source]
    for c in comps
        s = R * unit(c.ra, c.dec)
        dot(s, up) > 0 || continue
        for β in 1:meta.Nfreq
            I, Q, U, V = c.spectrum(ch[β])
            λ = SPEED_OF_LIGHT / ch[β]
            fringes = [cis(2π / λ * dot(pos[:, i], s - s0)) for i in 1:meta.Nant]
            for α in 1:meta.Nbase
                b = pos[:, bl[2, α]] .- pos[:, bl[1, α]]
                v = fringes[bl[1, α]] * conj(fringes[bl[2, α]])
                if c isa GPUGaussianSource
                    major, minor, wmaj, wmin = TTCalX._gaussian_axes(c, R)
                    v *= exp(-wmaj * (dot(b, major) / λ)^2 - wmin * (dot(b, minor) / λ)^2)
                end
                out.xx[α, β] += (I + Q) * v; out.xy[α, β] += complex(U, -V) * v
                out.yx[α, β] += complex(U, V) * v; out.yy[α, β] += (I - Q) * v
            end
        end
    end
    out
end

@testset "Exact sky model" begin
    meta = synthetic_meta()
    pl(I, V=0.0) = GPUPowerLaw(I, 0.1I, 0.05I, V, 50e6, [-0.7, -0.1])
    ra1, dec1 = azel_radec(60.0, 50.0)
    ra2, dec2 = azel_radec(200.0, 30.0)
    ra3, dec3 = azel_radec(100.0, -20.0)      # below the horizon
    multi = GPUMultiSource("multi", GPUSource[
        GPUPointSource("p", ra1, dec1, pl(1000.0, 3.0)),
        GPUGaussianSource("g", ra2, dec2, pl(500.0), deg2rad(0.5), deg2rad(0.2), deg2rad(30.0)),
        GPUPointSource("down", ra3, dec3, pl(1e6))])

    @testset "matches the reference loop (point, Gaussian, horizon)" begin
        vis = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(vis, meta, multi, TEST_MJD)
        # ITRF coordinates are ~6.4e6 m, so delays carry ~1e-9 m of rounding
        # and the two evaluation orders agree to ~1e-9 (≪ any model error).
        @test relmax(vis, reference_exact_model(meta, multi, TEST_MJD)) < 1e-8
    end

    @testset "accumulates into existing visibilities" begin
        vis = random_vis(meta.Nbase, meta.Nfreq)
        before = host_copy(vis)
        genvis_exact!(vis, meta, multi, TEST_MJD)
        ref = reference_exact_model(meta, multi, TEST_MJD)
        @test maximum(abs, vis.xx .- (before.xx .+ ref.xx)) < 1e-8 * maximum(abs, ref.xx)
    end

    @testset "point source at the phase centre" begin
        src = GPUPointSource("c", meta.phase_center_ra, meta.phase_center_dec, GPUPowerLaw(10.0, 1.0, 2.0, 3.0, 50e6, [0.0]))
        vis = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(vis, meta, src, TEST_MJD)
        @test all(vis.xx .≈ 11.0) && all(vis.yy .≈ 9.0)
        @test all(vis.xy .≈ complex(2.0, -3.0)) && all(vis.yx .≈ complex(2.0, 3.0))
    end

    @testset "below the horizon: no model" begin
        vis = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(vis, meta, GPUPointSource("down", ra3, dec3, pl(1e6)), TEST_MJD)
        @test all(iszero, vis.xx)
    end

    @testset "zero-width Gaussian equals a point" begin
        a = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        b = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
        genvis_exact!(a, meta, GPUGaussianSource("g0", ra2, dec2, pl(5.0), 0.0, 0.0, 0.3), TEST_MJD)
        genvis_exact!(b, meta, GPUPointSource("p0", ra2, dec2, pl(5.0)), TEST_MJD)
        @test relmax(a, b) < 1e-14
    end

    @testset "power law uses log10 (TTCal convention)" begin
        sp = GPUPowerLaw(100.0, 0.0, 0.0, 0.0, 10e6, [-1.0, 0.5])
        x = log10(2.0)
        @test sp(20e6)[1] ≈ 100.0 * 10^(-x + 0.5x^2)
    end

    if is_gpu_available()
        @testset "GPU kernel matches the host threads" begin
            gmeta = synthetic_meta(gpu=true)
            g = GPUVisibilities(gmeta.Nbase, gmeta.Nfreq; gpu=true)
            h = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=false)
            genvis_exact!(g, gmeta, multi, TEST_MJD)
            genvis_exact!(h, meta, multi, TEST_MJD)
            @test relmax(host_copy(g), h) < 1e-12
        end
    end
end
