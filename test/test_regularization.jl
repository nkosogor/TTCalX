@testset "Phase prior" begin
    @testset "tukey_prior" begin
        @test tukey_prior(0.1, 0.2, 0.5) == 1.0
        @test tukey_prior(0.7, 0.2, 0.5) == 0.0
        @test tukey_prior(0.45, 0.2, 0.5) ≈ 0.5
        ts = range(0.2, 0.7; length=50)
        @test issorted(tukey_prior.(ts, 0.2, 0.5); rev=true)        # monotonic taper
    end

    @testset "regularize_element" begin
        z = 2.0 * cis(1.0)
        @test regularize_element(z, 1.0, Inf, 0.5, 1.0) === z         # no prior for this gain
        @test regularize_element(z, 1.0, 0.2, 0.5, 0.0) === z         # λ = 0
        @test regularize_element(z, 1.0, 1.2, 0.5, 1.0) === z         # already inside the flat region
        r = regularize_element(z, 1.0, 0.2, 0.3, 10.0)
        @test 0.2 <= angle(r) < 0.5                                     # pulled into the taper
        @test abs(r) ≈ 2.0 * cos(1.0 - angle(r))                        # optimal modulus at that phase
        rneg = regularize_element(conj(z), 1.0, 0.2, 0.3, 10.0)
        @test rneg ≈ conj(r)                                            # symmetric in the phase sign
        weak = regularize_element(2.0 * cis(0.45), 1e6, 0.2, 0.3, 1e-6)
        @test abs(angle(weak) - 0.45) < 1e-3                            # a weak prior barely moves it
    end

    @testset "phase_regularization" begin
        meta = synthetic_meta()
        reg = phase_regularization(meta, 1.0, 1.0; gpu=false)
        @test size(reg.start) == (meta.Nant, meta.Nfreq)
        ρ = antenna_planar_distances(meta.antenna_positions)
        expected = 2π * ρ[3] * meta.channels[2] / SPEED_OF_LIGHT * sin(deg2rad(1.0))
        @test reg.start[3, 2] ≈ (expected >= π ? Inf : expected)
        @test 0.0 <= regularized_fraction(reg) <= 1.0
        free = phase_regularization(meta, 89.0, 1.0; gpu=false)
        @test regularized_fraction(free) == 0.0                         # φ_start ≥ π everywhere
        wide = phase_regularization(meta, 1.0, 1.0; gpu=false, nfreq=1)
        @test size(wide.start) == (meta.Nant, 1)
    end

    @testset "an inactive prior leaves the solve unchanged" begin
        meta, model, measured, _ = synthetic_problem(noise=1.0)
        free = phase_regularization(meta, 89.0, 1.0; gpu=false)
        a = identity_gains(meta); b = identity_gains(meta)
        solve_gains!(a, measured, model, meta; maxiter=5)
        solve_gains!(b, measured, model, meta; maxiter=5, reg=free)
        @test maximum(abs, a.xx .- b.xx) < 1e-12 * maximum(abs, a.xx)
    end

    @testset "a tight prior constrains the gain phases" begin
        # Damped half steps average the current and the regularized gains, so
        # with a flat region + taper below π/2 the phases stay inside it.
        meta, model, measured, _ = synthetic_problem()
        reg = phase_regularization(meta, 0.02, 0.02; gpu=false, lambda=1e6)
        @test maximum(filter(isfinite, Array(reg.start) .+ Array(reg.width))) < π / 2
        cal = identity_gains(meta)
        solve_gains!(cal, measured, model, meta; solver=:damped, maxiter=20, reg=reg)
        limit = Array(reg.start) .+ Array(reg.width)
        active = isfinite.(Array(reg.start))
        @test any(active)
        @test all(abs.(angle.(cal.xx[active])) .<= limit[active] .+ 1e-6)
        @test all(abs.(angle.(cal.yy[active])) .<= limit[active] .+ 1e-6)
    end
end
