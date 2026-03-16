#!/usr/bin/env julia
#==============================================================================#
#                    GPU Peel-and-Image Pipeline                                #
#==============================================================================#
#
# GPU-accelerated peel + image workflow for radio interferometry.
# Uses TTCalX's GPU peeling engine with the w-stacking imager.
#
# USAGE:
#   julia bin/peel_and_image_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...
#   julia bin/peel_and_image_gpu.jl image [options] <ms1> [ms2] ...
#   julia bin/peel_and_image_gpu.jl --demo
#
# COMMANDS:
#   peel    - Peel sources (diagonal Jones) + image residuals
#   zest    - Zest sources (full Jones) + image residuals
#   shave   - Shave sources (diagonal, wideband) + image residuals
#   prune   - Prune sources (full Jones, wideband) + image residuals
#   image   - Image only (no peeling, no sources.json needed)
#
# OPTIONS:
#   --maxiter=N       Max stefcal iterations per solve (default: 30)
#   --tolerance=T     Convergence tolerance (default: 1e-4)
#   --minuvw=M        Min baseline length in wavelengths (default: 10.0)
#   --peeliter=P      Number of peeling iterations (default: 3)
#   --column=COL      MS data column (default: CORRECTED_DATA)
#   --size=N          Image size in pixels (default: 1024, must be even)
#   --scale=DEG       Pixel size in degrees (default: auto)
#   --wlayers=N       Number of w-layers (default: auto)
#   --weight=SCHEME   Weighting: natural, uniform, briggs (default: natural)
#   --robust=R        Briggs robust parameter, -2 to 2 (default: 0)
#   --output=PREFIX   Output file prefix (default: image)
#   --verbose         Show detailed diagnostic output
#   --quiet           Suppress all output except errors
#   --help            Show this help message
#
# EXAMPLES:
#   julia bin/peel_and_image_gpu.jl peel sources.json data.ms
#   julia bin/peel_and_image_gpu.jl zest --size=2048 sources.json *.ms
#   julia bin/peel_and_image_gpu.jl image --scale=0.03 data.ms
#   julia bin/peel_and_image_gpu.jl --demo
#
#==============================================================================#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "TTCalX.jl"))
using .TTCalX
using .TTCalX.GPUImager
using Printf
using Statistics
using FFTW
using PyCall

#==============================================================================#
# Argument parsing
#==============================================================================#

function print_help()
    println("""
GPU Peel-and-Image: GPU-accelerated peeling + wide-field imaging

USAGE:
  julia bin/peel_and_image_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...
  julia bin/peel_and_image_gpu.jl image [options] <ms1> [ms2] ...
  julia bin/peel_and_image_gpu.jl --demo

COMMANDS:
  peel    Peel sources (diagonal Jones) + image residuals
  zest    Zest sources (full Jones) + image residuals
  shave   Shave sources (diagonal, wideband) + image residuals
  prune   Prune sources (full Jones, wideband) + image residuals
  image   Image only (no peeling, no sources.json needed)

IMAGING OPTIONS:
  --size=N          Image size in pixels [default: 1024, must be even]
  --scale=DEG       Pixel size in degrees [default: auto from UVW]
  --wlayers=N       Number of w-layers [default: auto from w-range]
  --weight=SCHEME   Weighting: natural, uniform, briggs [default: natural]
  --robust=R        Briggs robust parameter, -2 to 2 [default: 0]
  --output=PREFIX   Output file prefix [default: image]

CALIBRATION OPTIONS:
  --maxiter=N       Max stefcal iterations per solve [default: 30]
  --tolerance=T     Convergence tolerance [default: 1e-4]
  --minuvw=M        Min baseline length in wavelengths [default: 10.0]
  --peeliter=P      Number of peeling iterations [default: 3]
  --column=COL      MS data column to read/write [default: CORRECTED_DATA]
  --verbose         Show detailed diagnostic output
  --quiet           Suppress all output except errors
  --help            Show this help message

EXAMPLES:
  # Peel sources and image residuals
  julia bin/peel_and_image_gpu.jl peel sources.json data.ms

  # Zest with Briggs weighting
  julia bin/peel_and_image_gpu.jl zest --weight=briggs --robust=0.5 sources.json data.ms

  # Image only (no peeling)
  julia bin/peel_and_image_gpu.jl image --size=2048 data.ms

  # Run demo with synthetic data
  julia bin/peel_and_image_gpu.jl --demo

OUTPUT FILES:
  <prefix>_dirty.bin     Raw binary (Float64), loadable in Python/numpy
  <prefix>_dirty.csv     CSV file, loadable in DS9/topcat/Python
  <prefix>_dirty.pgm     ASCII PGM thumbnail for quick preview
""")
end

function parse_args(args)
    opts = Dict{String,Any}(
        "command" => "",
        "maxiter" => 30,
        "tolerance" => 1e-4,
        "minuvw" => 10.0,
        "peeliter" => 3,
        "column" => "CORRECTED_DATA",
        "verbosity" => :normal,
        "sources" => "",
        "ms_files" => String[],
        "image_size" => 1024,
        "cell_deg" => 0.0,
        "w_layers" => 0,
        "weighting" => :natural,
        "robust" => 0.0,
        "output" => "image"
    )
    
    positional = String[]
    
    for arg in args
        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        elseif arg == "--demo"
            opts["command"] = "demo"
            return opts
        elseif arg == "--quiet"
            opts["verbosity"] = :quiet
        elseif arg == "--verbose"
            opts["verbosity"] = :verbose
        elseif startswith(arg, "--maxiter=")
            opts["maxiter"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--tolerance=")
            opts["tolerance"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--minuvw=")
            opts["minuvw"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--peeliter=")
            opts["peeliter"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--column=")
            opts["column"] = String(split(arg, "=")[2])
        elseif startswith(arg, "--size=")
            opts["image_size"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--scale=")
            opts["cell_deg"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--wlayers=")
            opts["w_layers"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--weight=")
            opts["weighting"] = Symbol(split(arg, "=")[2])
        elseif startswith(arg, "--robust=")
            opts["robust"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--output=")
            opts["output"] = String(split(arg, "=")[2])
        elseif startswith(arg, "--")
            println("Unknown option: $arg")
            exit(1)
        else
            push!(positional, arg)
        end
    end
    
    if isempty(positional)
        println("Error: Need at least <command> and <ms_file>")
        println("Use --help for usage information")
        exit(1)
    end
    
    opts["command"] = positional[1]
    
    if opts["command"] == "image"
        # Image-only: no sources needed
        opts["ms_files"] = positional[2:end]
    else
        if length(positional) < 3
            println("Error: Need <command> <sources.json> <ms_file>")
            println("Use --help for usage information")
            exit(1)
        end
        opts["sources"] = positional[2]
        opts["ms_files"] = positional[3:end]
    end
    
    return opts
end

#==============================================================================#
# Output helpers
#==============================================================================#

"""Save a 2D Float64 array as raw binary (int32 header + Float64 data)."""
function save_bin(path::String, I::Matrix{Float64})
    open(path, "w") do f
        write(f, Int32(size(I, 1)))
        write(f, Int32(size(I, 2)))
        write(f, I)
    end
end

"""Save a rough ASCII PGM thumbnail for quick terminal viewing."""
function save_pgm(path::String, I::Matrix{Float64}; thumb_size::Int=64)
    N = size(I, 1)
    step = max(1, N ÷ thumb_size)
    thumb = I[1:step:end, 1:step:end]
    Nt = size(thumb, 1)
    lo, hi = minimum(thumb), maximum(thumb)
    if hi - lo < 1e-30
        pixels = zeros(UInt8, Nt, Nt)
    else
        pixels = UInt8.(clamp.(round.(Int, 255 .* (thumb .- lo) ./ (hi .- lo)), 0, 255))
    end
    open(path, "w") do f
        println(f, "P2")
        println(f, "$Nt $Nt")
        println(f, "255")
        for row in 1:Nt
            println(f, join(pixels[row, :], " "))
        end
    end
end

"""Save image as CSV."""
function save_csv(path::String, I::Matrix{Float64})
    open(path, "w") do f
        for row in 1:size(I, 1)
            println(f, join(I[row, :], ","))
        end
    end
end

"""Print image statistics."""
function image_stats(I::Matrix{Float64}, label::String)
    log_section(label)
    log_substep(@sprintf("Size       : %d x %d", size(I)...))
    log_substep(@sprintf("Min        : %+.6e Jy/beam", minimum(I)))
    log_substep(@sprintf("Max        : %+.6e Jy/beam", maximum(I)))
    log_substep(@sprintf("RMS        : %.6e Jy/beam", std(I)))
    log_substep(@sprintf("Mean       : %+.6e Jy/beam", mean(I)))
    peak = argmax(abs.(I))
    log_substep(@sprintf("Peak |I|   : %.6e  at pixel (%d, %d)", abs(I[peak]), peak[1], peak[2]))
    log_substep(@sprintf("Dynamic rng: %.1f  (peak / rms)", maximum(abs.(I)) / std(I)))
end

"""Save a FITS image with proper WCS headers (matches wsclean convention)."""
function save_fits(path::String, I::Matrix{Float64}, meta, config)
    astropy_io_fits = pyimport("astropy.io.fits")
    np_mod = pyimport("numpy")

    N = config.image_size
    cell_deg = rad2deg(config.cell_size)
    ra_deg  = rad2deg(meta.phase_center_ra)
    dec_deg = rad2deg(meta.phase_center_dec)

    # Compute reference frequency (band center, matching wsclean convention)
    channels_cpu = meta.channels isa CuArray ? Array(meta.channels) : meta.channels
    Nfreq = length(channels_cpu)
    if Nfreq > 1
        chan_width = channels_cpu[2] - channels_cpu[1]
        center_freq = channels_cpu[1] + (Nfreq - 1) / 2.0 * chan_width
        total_bw = Nfreq * chan_width
    else
        center_freq = channels_cpu[1]
        total_bw = channels_cpu[1]
    end

    # Orientation: transpose Julia [x,y] to FITS [y,x], then flipud (vertical flip)
    # Empirically verified to match wsclean pixel layout (corr=0.82 with NN gridding)
    img_np = np_mod.array(reverse(permutedims(I), dims=1), dtype=np_mod.float64)
    img_4d = np_mod.reshape(img_np, (1, 1, N, N))

    hdu = astropy_io_fits.PrimaryHDU(data=img_4d)
    hdr = hdu.header

    # RA axis (axis 1 in FITS = last numpy axis)
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CRPIX1"] = Float64(N ÷ 2 + 1)
    hdr["CDELT1"] = -cell_deg            # RA increases to the left
    hdr["CRVAL1"] = ra_deg
    hdr["CUNIT1"] = "deg"

    # Dec axis (axis 2)
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRPIX2"] = Float64(N ÷ 2 + 1)
    hdr["CDELT2"] = cell_deg
    hdr["CRVAL2"] = dec_deg
    hdr["CUNIT2"] = "deg"

    # Frequency axis (axis 3) — band center + total bandwidth
    hdr["CTYPE3"] = "FREQ"
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = total_bw
    hdr["CRVAL3"] = center_freq
    hdr["CUNIT3"] = "Hz"
    hdr["SPECSYS"] = "TOPOCENT"

    # Stokes axis (axis 4), Stokes I = 1
    hdr["CTYPE4"] = "STOKES"
    hdr["CRPIX4"] = 1.0
    hdr["CDELT4"] = 1.0
    hdr["CRVAL4"] = 1.0      # 1 = Stokes I
    hdr["CUNIT4"] = ""

    # Standard FITS keywords
    hdr["BUNIT"]  = "JY/BEAM"
    hdr["BTYPE"]  = "Intensity"
    hdr["ORIGIN"] = "TTCalX"
    hdr["TELESCOP"] = "OVRO-LWA"
    hdr["EQUINOX"] = 2000.0
    hdr["LONPOLE"] = 180.0

    hdu.writeto(path, overwrite=true)
end

"""Save all output files for an image."""
function save_image_outputs(I::Matrix{Float64}, prefix::String;
                            meta=nothing, config=nothing)
    save_bin("$(prefix).bin", I)
    save_pgm("$(prefix).pgm", I)
    save_csv("$(prefix).csv", I)
    formats = ".bin, .pgm, .csv"
    if meta !== nothing && config !== nothing
        try
            save_fits("$(prefix).fits", I, meta, config)
            formats *= ", .fits"
        catch e
            log_warning("FITS save failed: $e (install astropy?)")
        end
    end
    log_substep("Saved: $(prefix)$(formats)")
end

#==============================================================================#
# Main processing
#==============================================================================#

function process_ms(ms_path::String, opts, sources)
    t_start = time()
    command = opts["command"]
    image_only = (command == "image")
    
    # Read MS
    log_step("Reading MS: $(basename(ms_path))...")
    vis, cal, meta, baseline_dict, Nrows = read_ms_to_gpu(
        ms_path; gpu=true, column=opts["column"]
    )
    log_substep("Antennas: $(meta.Nant), Baselines: $(meta.Nbase), Channels: $(meta.Nfreq)")
    
    # Configure imager
    log_step("Configuring imager...")
    cell_size = opts["cell_deg"] > 0 ? deg2rad(opts["cell_deg"]) : nothing
    
    if cell_size !== nothing && opts["w_layers"] > 0
        config = GPUImagerConfig(
            image_size = opts["image_size"],
            cell_size  = cell_size,
            w_layers   = opts["w_layers"],
            weighting  = opts["weighting"],
            robust     = opts["robust"]
        )
    else
        config = auto_configure_imager(meta, vis;
                                        image_size=opts["image_size"],
                                        cell_size=cell_size)
        config = GPUImagerConfig(
            image_size = config.image_size,
            cell_size  = config.cell_size,
            w_layers   = opts["w_layers"] > 0 ? opts["w_layers"] : config.w_layers,
            weighting  = opts["weighting"],
            robust     = opts["robust"]
        )
    end
    
    log_substep(@sprintf("Image: %d x %d, cell=%.2f arcsec, FoV=%.2f deg",
                config.image_size, config.image_size,
                rad2deg(config.cell_size) * 3600, rad2deg(field_of_view(config))))
    log_substep(@sprintf("W-layers: %d, Weighting: %s", config.w_layers, config.weighting))
    
    prefix = opts["output"]
    
    if image_only
        # Image only (no peeling)
        log_step("Imaging (no peeling)...")
        t0 = time()
        img = make_image(vis, meta, config)
        dt = time() - t0
        
        I = img.stokes_I isa CuArray ? Array(img.stokes_I) : img.stokes_I
        image_stats(I, "Dirty Image")
        save_image_outputs(I, "$(prefix)_dirty"; meta=meta, config=config)
        log_substep(@sprintf("Imaging took %.2f s", dt))
    else
        # Image BEFORE peeling
        log_step("Imaging BEFORE peeling...")
        t0 = time()
        img_before = make_image(vis, meta, config)
        dt = time() - t0
        
        I_before = img_before.stokes_I isa CuArray ? Array(img_before.stokes_I) : img_before.stokes_I
        image_stats(I_before, "Dirty Image BEFORE Peeling")
        save_image_outputs(I_before, "$(prefix)_before"; meta=meta, config=config)
        log_substep(@sprintf("Imaging took %.2f s", dt))
        
        # Peel sources
        log_step("Peeling $(length(sources)) source(s) ($(opts["peeliter"]) iterations)...")
        t0 = time()
        calibrations = peel_gpu!(vis, meta, sources;
                                 peeliter=opts["peeliter"],
                                 maxiter=opts["maxiter"],
                                 tolerance=opts["tolerance"],
                                 minuvw=opts["minuvw"],
                                 phase_center_ra=meta.phase_center_ra,
                                 phase_center_dec=meta.phase_center_dec,
                                 lst=0.0)
        dt = time() - t0
        log_substep(@sprintf("Peeling took %.2f s", dt))
        
        # Image AFTER peeling
        log_step("Imaging AFTER peeling...")
        t0 = time()
        img_after = make_image(vis, meta, config)
        dt = time() - t0
        
        I_after = img_after.stokes_I isa CuArray ? Array(img_after.stokes_I) : img_after.stokes_I
        image_stats(I_after, "Dirty Image AFTER Peeling")
        save_image_outputs(I_after, "$(prefix)_after"; meta=meta, config=config)
        log_substep(@sprintf("Imaging took %.2f s", dt))
        
        # Comparison
        log_section("Comparison")
        peak_before = maximum(abs.(I_before))
        peak_after = maximum(abs.(I_after))
        rms_before = std(I_before)
        rms_after = std(I_after)
        log_substep(@sprintf("Peak: %.4e -> %.4e  (%.1fx reduction)",
                    peak_before, peak_after, peak_before / max(peak_after, 1e-30)))
        log_substep(@sprintf("RMS:  %.4e -> %.4e  (%.1fx reduction)",
                    rms_before, rms_after, rms_before / max(rms_after, 1e-30)))
        log_substep(@sprintf("DR:   %.0f -> %.0f",
                    peak_before/rms_before, peak_after/rms_after))
        
        # Write back peeled data to MS
        log_step("Writing calibrated data to MS...")
        write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column=opts["column"])
    end
    
    t_elapsed = time() - t_start
    log_success(@sprintf("Completed in %.2f seconds", t_elapsed))
    return t_elapsed
end

#==============================================================================#
# Demo with synthetic data
#==============================================================================#

function demo()
    log_header("GPU Peel-and-Image Demo (Synthetic Data)")
    
    use_gpu = is_gpu_available()
    log_step("Using: $(use_gpu ? "GPU" : "CPU")")
    
    # Create synthetic observation
    log_step("Creating synthetic observation...")
    Nant = 16
    Nfreq = 32
    Nb = Nant * (Nant - 1) ÷ 2
    
    # Antenna positions (Y configuration)
    antenna_positions = zeros(Float64, 3, Nant)
    for i in 1:Nant
        arm = mod(i - 1, 3)
        station = (i - 1) ÷ 3
        spacing = 300.0
        
        if arm == 0
            antenna_positions[2, i] = station * spacing
        elseif arm == 1
            antenna_positions[1, i] = -station * spacing * cos(deg2rad(30))
            antenna_positions[2, i] = -station * spacing * sin(deg2rad(30))
        else
            antenna_positions[1, i] = station * spacing * cos(deg2rad(30))
            antenna_positions[2, i] = -station * spacing * sin(deg2rad(30))
        end
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
    
    log_substep("Antennas: $Nant, Baselines: $Nb, Frequencies: $Nfreq")
    
    # Create visibilities with two point sources
    log_step("Creating synthetic visibilities with 2 sources...")
    vis = GPUVisibilities(Nb, Nfreq, gpu=use_gpu)
    
    src1_l, src1_m, src1_flux = 0.01, 0.02, 100.0
    src2_l, src2_m, src2_flux = 0.05, -0.03, 30.0
    
    vis_xx = zeros(ComplexF64, Nb, Nfreq)
    c = 299792458.0
    for β in 1:Nfreq
        λ = c / channels[β]
        for α in 1:Nb
            u, v, w = uvw[1, α], uvw[2, α], uvw[3, α]
            n1 = sqrt(1 - src1_l^2 - src1_m^2)
            phase1 = -2π * (u*src1_l + v*src1_m + w*(n1-1)) / λ
            vis_xx[α, β] += src1_flux * exp(im * phase1)
            n2 = sqrt(1 - src2_l^2 - src2_m^2)
            phase2 = -2π * (u*src2_l + v*src2_m + w*(n2-1)) / λ
            vis_xx[α, β] += src2_flux * exp(im * phase2)
            vis_xx[α, β] += 1.0 * randn(ComplexF64)
        end
    end
    
    if use_gpu
        copyto!(vis.xx, CuArray(vis_xx))
        copyto!(vis.yy, CuArray(vis_xx))
    else
        copyto!(vis.xx, vis_xx)
        copyto!(vis.yy, vis_xx)
    end
    
    log_substep(@sprintf("Source 1: flux=%.0f, (l,m)=(%.3f, %.3f)", src1_flux, src1_l, src1_m))
    log_substep(@sprintf("Source 2: flux=%.0f, (l,m)=(%.3f, %.3f)", src2_flux, src2_l, src2_m))
    
    # Create source model for peeling
    spectrum = GPUPowerLaw(src1_flux, 0.0, 0.0, 0.0, 150e6, [-0.7])
    source = GPUPointSource("BrightSource", phase_center_ra + src1_l, phase_center_dec + src1_m, spectrum)
    sources = [GPUPeelingSource(source)]
    
    # Configure imager
    config = GPUImagerConfig(
        image_size=256,
        cell_size=deg2rad(1.0/60.0),
        w_layers=1,
        weighting=:natural
    )
    
    # Image BEFORE peeling
    log_step("Imaging BEFORE peeling...")
    img_before = make_image(vis, meta, config)
    I_before = img_before.stokes_I isa CuArray ? Array(img_before.stokes_I) : img_before.stokes_I
    log_substep(@sprintf("Peak: %.2f at pixel (%d, %d)", maximum(I_before), argmax(I_before)[1], argmax(I_before)[2]))
    
    # Peel and image
    log_step("Peeling source and imaging residuals...")
    img_after = peel_and_image(vis, meta, sources, config;
                               peeliter=3, maxiter=20, tolerance=1e-3,
                               phase_center_ra=phase_center_ra, 
                               phase_center_dec=phase_center_dec,
                               lst=0.0)
    
    I_after = img_after.stokes_I isa CuArray ? Array(img_after.stokes_I) : img_after.stokes_I
    log_substep(@sprintf("Peak after peeling: %.2f", maximum(I_after)))
    log_substep(@sprintf("RMS: %.4f", std(I_after)))
    
    reduction = maximum(I_before) / maximum(I_after)
    log_substep(@sprintf("Peak reduction: %.1fx", reduction))
    
    if reduction > 2.0
        log_success("Peeling successfully reduced bright source!")
    else
        log_warning("Peeling may not have fully converged")
    end
    
    log_success("Demo complete!")
end

#==============================================================================#
# Entry point
#==============================================================================#

function main()
    if length(ARGS) == 0 || ARGS[1] == "--help" || ARGS[1] == "-h"
        print_help()
        exit(0)
    end
    
    opts = parse_args(ARGS)
    
    if opts["command"] == "demo"
        demo()
        return
    end
    
    command = opts["command"]
    set_verbosity(opts["verbosity"])
    
    log_header("GPU Peel-and-Image: $(uppercase(command))")
    
    # Select wrapping function for sources
    wrap_func = if command == "peel"
        GPUPeelingSource
    elseif command == "zest"
        GPUZestingSource
    elseif command == "shave"
        GPUShavingSource
    elseif command == "prune"
        GPUPruningSource
    elseif command == "image"
        nothing  # No wrapping needed
    else
        log_error("Unknown command '$command'")
        println("Valid commands: peel, zest, shave, prune, image")
        exit(1)
    end
    
    # Initialize python-casacore
    log_section("Initialization")
    log_step("Loading python-casacore...")
    if !init_pycasacore()
        log_error("python-casacore not available")
        println("Install with: pip install python-casacore")
        exit(1)
    end
    log_success("python-casacore loaded")
    
    # Load sources (unless image-only mode)
    sources = []
    if command != "image"
        log_step("Loading sources from $(opts["sources"])...")
        sources_raw = read_gpu_sources(opts["sources"])
        sources = [wrap_func(s) for s in sources_raw]
        log_substep("Loaded $(length(sources)) sources")
        @verbose for s in sources
            log_detail("$(get_name(s))")
        end
    end
    
    # Show configuration
    log_section("Configuration")
    log_config(
        "Command" => command,
        "Image size" => "$(opts["image_size"]) x $(opts["image_size"])",
        "Weighting" => string(opts["weighting"]),
        "Sources" => command == "image" ? "N/A" : opts["sources"],
        "Column" => opts["column"],
        "Max iter" => opts["maxiter"],
        "Min UVW" => "$(opts["minuvw"]) λ",
        "Peel iter" => opts["peeliter"],
        "MS files" => length(opts["ms_files"])
    )
    
    # Process each MS file
    times = Float64[]
    ms_files = opts["ms_files"]
    
    for (i, ms_path) in enumerate(ms_files)
        log_section("[$i/$(length(ms_files))] $(basename(ms_path))")
        t = process_ms(ms_path, opts, sources)
        push!(times, t)
    end
    
    # Summary
    if length(ms_files) > 0
        log_section("Summary")
        for (i, t) in enumerate(times)
            note = i == 1 && length(times) > 1 ? " (includes JIT)" : ""
            log_substep(@sprintf("%-40s  %.2fs%s", basename(ms_files[i]), t, note))
        end
        if length(times) > 1
            avg_time = sum(times[2:end]) / length(times[2:end])
            log_substep(@sprintf("Average (excl. JIT): %.2f seconds", avg_time))
        end
        log_success("All done!")
    end
end

main()
