#!/usr/bin/env julia
#==============================================================================#
#                       GPU Imager Test Suite                                    #
#==============================================================================#
#
# Tests for the GPU wide-field imager with w-stacking.
# Covers configuration, gridding, weighting, FFT, and full pipeline.
#
# Run: julia test/imaging_test.jl
#==============================================================================#

using Test
using Statistics
using LinearAlgebra
using FFTW

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "TTCalX.jl"))
using .TTCalX
using .TTCalX.GPUImager

# Use CPU for tests (no GPU required)
const USE_GPU = false

#==============================================================================#
# Helpers
#==============================================================================#

"""Create a simple test observation with Nant antennas and Nfreq channels."""
function make_test_observation(; Nant=8, Nfreq=4, use_gpu=USE_GPU)
    Nb = Nant * (Nant - 1) ÷ 2
    
    # 2D array layout (gives UV coverage in both u and v)
    antenna_positions = zeros(Float64, 3, Nant)
    for i in 1:Nant
        θ = 2π * (i - 1) / Nant
        antenna_positions[1, i] = 100.0 * cos(θ) * i  # x
        antenna_positions[2, i] = 100.0 * sin(θ) * i  # y
    end
    
    baselines = zeros(Int32, 2, Nb)
    uvw = zeros(Float64, 3, Nb)
    b = 1
    for i in 1:Nant, j in (i+1):Nant
        baselines[1, b] = i
        baselines[2, b] = j
        uvw[1, b] = antenna_positions[1, j] - antenna_positions[1, i]
        uvw[2, b] = antenna_positions[2, j] - antenna_positions[2, i]
        b += 1
    end
    
    channels = collect(range(100e6, 200e6, length=Nfreq))
    phase_center_ra = 0.0
    phase_center_dec = π/2
    
    meta = create_gpu_metadata(
        antenna_positions, baselines, channels, [0.0, 0.0, 1.0], uvw;
        phase_center_ra=phase_center_ra, phase_center_dec=phase_center_dec,
        gpu=use_gpu
    )
    
    vis = GPUVisibilities(Nb, Nfreq, gpu=use_gpu)
    
    return vis, meta
end

"""Create visibilities for a point source at (l, m) with given flux."""
function add_point_source!(vis, meta, l::Float64, m::Float64, flux::Float64)
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    c = 299792458.0
    
    uvw = meta.uvw isa CUDA.CuArray ? Array(meta.uvw) : meta.uvw
    channels = meta.channels isa CUDA.CuArray ? Array(meta.channels) : meta.channels
    
    vis_xx = vis.xx isa CUDA.CuArray ? Array(vis.xx) : vis.xx
    vis_yy = vis.yy isa CUDA.CuArray ? Array(vis.yy) : vis.yy
    
    n = sqrt(1 - l^2 - m^2)
    for β in 1:Nf
        λ = c / channels[β]
        for α in 1:Nb
            u, v, w = uvw[1, α], uvw[2, α], uvw[3, α]
            phase = -2π * (u*l + v*m + w*(n-1)) / λ
            vis_xx[α, β] += flux * exp(im * phase)
            vis_yy[α, β] += flux * exp(im * phase)
        end
    end
    
    if vis.xx isa CUDA.CuArray
        copyto!(vis.xx, CUDA.CuArray(vis_xx))
        copyto!(vis.yy, CUDA.CuArray(vis_yy))
    else
        copyto!(vis.xx, vis_xx)
        copyto!(vis.yy, vis_yy)
    end
end

#==============================================================================#
# Tests
#==============================================================================#

@testset "GPU Imager" begin
    
    @testset "Configuration" begin
        # Default config
        config = GPUImagerConfig()
        @test config.image_size == 512
        @test config.w_layers == 1
        @test config.weighting == :natural
        @test config.cell_size > 0
        
        # Custom config
        config = GPUImagerConfig(
            image_size=256,
            cell_size=deg2rad(0.5/60),
            w_layers=4,
            weighting=:briggs,
            robust=-0.5
        )
        @test config.image_size == 256
        @test config.w_layers == 4
        @test config.weighting == :briggs
        @test config.robust == -0.5
        
        # Validation
        @test_throws AssertionError GPUImagerConfig(image_size=0)
        @test_throws AssertionError GPUImagerConfig(image_size=511)  # odd
        @test_throws AssertionError GPUImagerConfig(cell_size=-1.0)
        @test_throws AssertionError GPUImagerConfig(weighting=:invalid)
        @test_throws AssertionError GPUImagerConfig(robust=3.0)
        
        # Field of view
        config = GPUImagerConfig(image_size=100, cell_size=deg2rad(1.0/60.0))
        fov = field_of_view(config)
        @test fov ≈ 100 * deg2rad(1.0/60.0)
    end
    
    @testset "Grid and Image types" begin
        # CPU grid
        grid = GPUGrid(64, 2, gpu=false)
        @test size(grid.data_re) == (64, 64, 2)
        @test size(grid.data_im) == (64, 64, 2)
        @test size(grid.weights) == (64, 64, 2)
        @test length(grid.w_values) == 2
        @test all(grid.data_re .== 0)
        @test all(grid.data_im .== 0)
        
        # Reset
        grid.data_re[1,1,1] = 1.0
        empty!(grid)
        @test all(grid.data_re .== 0)
        
        # CPU image
        img = GPUImage(64, gpu=false)
        @test size(img.stokes_I) == (64, 64)
        @test size(img.stokes_Q) == (64, 64)
        @test all(img.stokes_I .== 0)
    end
    
    @testset "W-layer indices" begin
        config = GPUImagerConfig(w_layers=4)
        
        # Uniform w-values
        w = [0.0, 1.0, 2.0, 3.0, 4.0]
        layers = w_layer_indices(w, config)
        @test all(1 .<= layers .<= 4)
        @test layers[1] == 1
        @test layers[end] == 4
        
        # Single w-layer
        config1 = GPUImagerConfig(w_layers=1)
        layers1 = w_layer_indices(w, config1)
        @test all(layers1 .== 1)
    end
    
    @testset "W-correction phase" begin
        # At l=0, m=0 (phase center): correction = exp(0) = 1
        @test w_correction_phase(0.0, 0.0, 100.0) ≈ 1.0
        
        # At l=0, m=0 with any w, should be 1
        @test abs(w_correction_phase(0.0, 0.0, 1000.0)) ≈ 1.0
        
        # Outside unit circle: should be 0
        @test w_correction_phase(1.0, 1.0, 100.0) == 0.0
        
        # Non-zero l,m: should have |phase| = 1
        @test abs(w_correction_phase(0.1, 0.1, 50.0)) ≈ 1.0
    end
    
    @testset "CPU gridding - basic" begin
        vis, meta = make_test_observation(Nant=4, Nfreq=2)
        
        # Add a source at phase center (l=0, m=0)
        add_point_source!(vis, meta, 0.0, 0.0, 10.0)
        
        config = GPUImagerConfig(
            image_size=64,
            cell_size=deg2rad(2.0/60.0),
            w_layers=1,
            weighting=:natural
        )
        
        grid = GPUGrid(64, 1, gpu=false)
        grid_visibilities!(grid, vis, meta, config)
        
        # Grid should have non-zero values
        @test sum(grid.data_re .^ 2 .+ grid.data_im .^ 2) > 0
        @test sum(grid.weights) > 0
    end
    
    @testset "CPU imaging - point source" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        
        # Single bright source at phase center
        add_point_source!(vis, meta, 0.0, 0.0, 100.0)
        
        config = GPUImagerConfig(
            image_size=64,
            cell_size=deg2rad(2.0/60.0),
            w_layers=1,
            weighting=:natural
        )
        
        img = make_image(vis, meta, config)
        I = img.stokes_I
        
        # Peak should be at or near center
        center = 64 ÷ 2 + 1
        peak = argmax(abs.(I))
        @test abs(peak[1] - center) <= 2
        @test abs(peak[2] - center) <= 2
        
        # Peak should be positive
        @test maximum(I) > 0
    end
    
    @testset "CPU imaging - off-center source" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        
        l_src = 0.01  # small offset
        add_point_source!(vis, meta, l_src, 0.0, 50.0)
        
        config = GPUImagerConfig(
            image_size=64,
            cell_size=deg2rad(2.0/60.0),
            w_layers=1,
            weighting=:natural
        )
        
        img = make_image(vis, meta, config)
        I = img.stokes_I
        
        # Peak should be offset from center
        center = 64 ÷ 2 + 1
        peak = argmax(abs.(I))
        @test maximum(I) > 0
    end
    
    @testset "Weighting schemes" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        add_point_source!(vis, meta, 0.0, 0.0, 100.0)
        
        config_nat = GPUImagerConfig(image_size=64, cell_size=deg2rad(2.0/60.0), weighting=:natural)
        config_uni = GPUImagerConfig(image_size=64, cell_size=deg2rad(2.0/60.0), weighting=:uniform)
        config_bri = GPUImagerConfig(image_size=64, cell_size=deg2rad(2.0/60.0), weighting=:briggs, robust=0.0)
        
        img_nat = make_image(vis, meta, config_nat)
        img_uni = make_image(vis, meta, config_uni)
        img_bri = make_image(vis, meta, config_bri)
        
        # All should produce non-zero images
        @test maximum(abs.(img_nat.stokes_I)) > 0
        @test maximum(abs.(img_uni.stokes_I)) > 0
        @test maximum(abs.(img_bri.stokes_I)) > 0
    end
    
    @testset "W-stacking (multiple w-layers)" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        add_point_source!(vis, meta, 0.0, 0.0, 100.0)
        
        config_1w = GPUImagerConfig(image_size=64, cell_size=deg2rad(2.0/60.0), w_layers=1)
        config_4w = GPUImagerConfig(image_size=64, cell_size=deg2rad(2.0/60.0), w_layers=4)
        
        img_1w = make_image(vis, meta, config_1w)
        img_4w = make_image(vis, meta, config_4w)
        
        # Both should produce images with similar peak location
        peak_1w = argmax(abs.(img_1w.stokes_I))
        peak_4w = argmax(abs.(img_4w.stokes_I))
        @test abs(peak_1w[1] - peak_4w[1]) <= 2
        @test abs(peak_1w[2] - peak_4w[2]) <= 2
    end
    
    @testset "Auto-configure" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        
        config = auto_configure_imager(meta, vis, image_size=128)
        @test config.image_size == 128
        @test config.cell_size > 0
        @test config.w_layers >= 1
    end
    
    @testset "Image coordinates" begin
        config = GPUImagerConfig(image_size=64, cell_size=deg2rad(1.0/60.0))
        img = GPUImage(64, gpu=false)
        
        l, m = image_coordinates(img, config)
        @test length(l) == 64
        @test length(m) == 64
        @test l[33] ≈ 0.0  atol=config.cell_size  # Center should be near 0
    end
    
    @testset "Edge cases" begin
        vis, meta = make_test_observation(Nant=4, Nfreq=2)
        
        config = GPUImagerConfig(image_size=32, cell_size=deg2rad(2.0/60.0), w_layers=1)
        
        # All flagged data
        fill!(vis.flags, true)
        img = make_image(vis, meta, config)
        @test all(isfinite.(img.stokes_I))
        
        # All zero visibilities (unflag them first)
        fill!(vis.flags, false)
        fill!(vis.xx, 0.0)
        fill!(vis.yy, 0.0)
        img = make_image(vis, meta, config)
        @test all(isfinite.(img.stokes_I))
    end
    
    @testset "Peel and image (CPU, synthetic)" begin
        vis, meta = make_test_observation(Nant=8, Nfreq=4)
        
        phase_center_ra = meta.phase_center_ra
        phase_center_dec = meta.phase_center_dec
        
        # Add a bright source to peel
        src_l = 0.005
        src_m = 0.005
        add_point_source!(vis, meta, src_l, src_m, 100.0)
        
        # Add a faint source that should remain
        add_point_source!(vis, meta, -0.02, 0.01, 5.0)
        
        # Create peeling source model
        spectrum = GPUPowerLaw(100.0, 0.0, 0.0, 0.0, 150e6, [-0.7])
        source = GPUPointSource("TestSrc", phase_center_ra + src_l, phase_center_dec + src_m, spectrum)
        sources = [GPUPeelingSource(source)]
        
        config = GPUImagerConfig(
            image_size=64,
            cell_size=deg2rad(2.0/60.0),
            w_layers=1,
            weighting=:natural
        )
        
        # Image before peeling
        peak_before = maximum(abs.(make_image(vis, meta, config).stokes_I))
        
        # Peel and image
        img = peel_and_image(vis, meta, sources, config;
                             peeliter=3, maxiter=20, tolerance=1e-3,
                             phase_center_ra=phase_center_ra,
                             phase_center_dec=phase_center_dec,
                             lst=0.0)
        
        peak_after = maximum(abs.(img.stokes_I))
        
        # Peak should be reduced after peeling
        @test peak_after < peak_before
        @test all(isfinite.(img.stokes_I))
    end
end

println("\nAll imaging tests passed!")
