#!/usr/bin/env julia
#==============================================================================#
#                          GPU-TTCal Command Line Interface                     #
#==============================================================================#
#
# GPU-accelerated direction-dependent calibration for radio interferometry.
#
# USAGE:
#   julia bin/ttcal_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...
#
# COMMANDS:
#   peel   - Diagonal Jones matrices, per frequency channel
#   zest   - Full Jones matrices, per frequency channel
#   shave  - Diagonal Jones matrices, wideband (one per subband)
#   prune  - Full Jones matrices, wideband (one per subband)
#
# OPTIONS:
#   --maxiter=N       Max stefcal iterations per solve (default: 30)
#   --tolerance=T     Convergence tolerance (default: 1e-4)
#   --minuvw=M        Min baseline length in wavelengths (default: 10.0)
#   --peeliter=P      Number of peeling iterations (default: 3)
#   --column=COL      MS data column (default: CORRECTED_DATA)
#   --verbose         Show detailed diagnostic output
#   --quiet           Suppress all output except errors
#   --help            Show this help message
#
# EXAMPLES:
#   julia bin/ttcal_gpu.jl peel sources.json data.ms
#   julia bin/ttcal_gpu.jl zest sources.json *.ms --maxiter=50
#   julia bin/ttcal_gpu.jl shave --minuvw=15 sources.json B*.ms
#
#==============================================================================#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "gpu", "GPUTTCal.jl"))
using .GPUTTCal
using Printf

#==============================================================================#
# Argument parsing
#==============================================================================#

function print_help()
    println("""
GPU-TTCal: GPU-accelerated direction-dependent calibration

USAGE:
  julia bin/ttcal_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...

COMMANDS:
  peel    Diagonal Jones calibration, per frequency channel
  zest    Full Jones calibration, per frequency channel  
  shave   Diagonal Jones calibration, wideband (one per subband)
  prune   Full Jones calibration, wideband (one per subband)

OPTIONS:
  --maxiter=N       Max stefcal iterations per solve [default: 30]
  --tolerance=T     Convergence tolerance [default: 1e-4]
  --minuvw=M        Min baseline length in wavelengths [default: 10.0]
  --peeliter=P      Number of peeling iterations [default: 3]
  --column=COL      MS data column to read/write [default: CORRECTED_DATA]
  --verbose         Show detailed diagnostic output  
  --quiet           Suppress all output except errors
  --help            Show this help message

EXAMPLES:
  # Peel sources from a single MS
  julia bin/ttcal_gpu.jl peel sources.json data.ms
  
  # Zest multiple MS files (batch processing)
  julia bin/ttcal_gpu.jl zest sources.json *.ms
  
  # Peel with custom parameters
  julia bin/ttcal_gpu.jl peel --maxiter=50 --minuvw=15 sources.json data.ms

CALIBRATION MODES:
  peel/shave  Use diagonal Jones matrices (2 parameters per antenna)
              Good for phase-only or amplitude+phase calibration
              
  zest/prune  Use full Jones matrices (4 parameters per antenna)
              Handles polarization leakage and rotation
              
  peel/zest   Per-channel calibration (more degrees of freedom)
  shave/prune Wideband calibration (one solution per subband)

SOURCES FILE FORMAT:
  JSON array of source objects. Each source has:
    - name: source name
    - ra: right ascension (degrees or "HH:MM:SS")
    - dec: declination (degrees or "+DD:MM:SS")
    - flux: Stokes I flux in Jy (or spectrum object)
    - type: "point" or "gaussian"
    - major_fwhm, minor_fwhm, position_angle: for Gaussian sources

  Example sources.json:
  [
    {"name": "CasA", "ra": "23:23:24", "dec": "+58:48:54", "flux": 16530},
    {"name": "CygA", "ra": 299.868, "dec": 40.734, "flux": 16300}
  ]
""")
end

function parse_args(args)
    # Defaults
    opts = Dict{String,Any}(
        "command" => "",
        "maxiter" => 30,
        "tolerance" => 1e-4,
        "minuvw" => 10.0,
        "peeliter" => 3,
        "column" => "CORRECTED_DATA",
        "verbosity" => :normal,
        "sources" => "",
        "ms_files" => String[]
    )
    
    positional = String[]
    
    for arg in args
        if arg == "--help" || arg == "-h"
            print_help()
            exit(0)
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
            opts["column"] = split(arg, "=")[2]
        elseif startswith(arg, "--")
            println("Unknown option: $arg")
            exit(1)
        else
            push!(positional, arg)
        end
    end
    
    if length(positional) < 3
        println("Error: Need at least <command> <sources.json> <ms_file>")
        println("Use --help for usage information")
        exit(1)
    end
    
    opts["command"] = positional[1]
    opts["sources"] = positional[2]
    opts["ms_files"] = positional[3:end]
    
    return opts
end

#==============================================================================#
# Main processing
#==============================================================================#

function main()
    if length(ARGS) == 0 || ARGS[1] == "--help" || ARGS[1] == "-h"
        print_help()
        exit(0)
    end
    
    opts = parse_args(ARGS)
    
    command = opts["command"]
    
    # Set global verbosity
    set_verbosity(opts["verbosity"])
    
    # Select processing function
    process_func = if command == "peel"
        peel_gpu!
    elseif command == "zest"
        zest_gpu!
    elseif command == "shave"
        shave_gpu!
    elseif command == "prune"
        prune_gpu!
    else
        log_error("Unknown command '$command'")
        println("Valid commands: peel, zest, shave, prune")
        exit(1)
    end
    
    log_header("GPU-TTCal: $(uppercase(command))")
    
    log_section("Configuration")
    log_config(
        "Command" => command,
        "Sources" => opts["sources"],
        "Column" => opts["column"],
        "Max iter" => opts["maxiter"],
        "Tolerance" => opts["tolerance"],
        "Min UVW" => "$(opts["minuvw"]) λ",
        "Peel iter" => opts["peeliter"],
        "MS files" => length(opts["ms_files"])
    )
    
    # Initialize python-casacore
    log_section("Initialization")
    log_step("Loading python-casacore...")
    
    if !init_pycasacore()
        log_error("python-casacore not available")
        println("Install with: pip install python-casacore")
        exit(1)
    end
    log_success("python-casacore loaded")
    
    # Load sources
    log_step("Loading sources from $(opts["sources"])...")
    sources = read_gpu_sources(opts["sources"])
    log_substep("Loaded $(length(sources)) sources")
    @verbose for s in sources
        log_detail("$(get_name(s))")
    end
    
    # Process each MS file
    times = Float64[]
    ms_files = opts["ms_files"]
    
    for (i, ms_path) in enumerate(ms_files)
        log_section("[$i/$(length(ms_files))] $(basename(ms_path))")
        
        t_start = time()
        
        # Read MS
        log_step("Reading MS data...")
        vis, cal, meta, baseline_dict, Nrows = read_ms_to_gpu(
            ms_path; gpu=true, column=opts["column"]
        )
        
        if i == 1
            log_substep("Antennas: $(meta.Nant), Baselines: $(meta.Nbase), Channels: $(meta.Nfreq)")
        end
        
        # Run calibration
        process_func(vis, meta, sources;
            maxiter=opts["maxiter"],
            tolerance=opts["tolerance"],
            minuvw=opts["minuvw"],
            peeliter=opts["peeliter"],
            phase_center_ra=meta.phase_center_ra,
            phase_center_dec=meta.phase_center_dec,
            lst=0.0
        )
        
        # Write back
        log_step("Writing calibrated data...")
        write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column=opts["column"])
        
        t_elapsed = time() - t_start
        push!(times, t_elapsed)
        
        log_success(@sprintf("Completed in %.2f seconds", t_elapsed))
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
