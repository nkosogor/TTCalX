#!/usr/bin/env julia
#==============================================================================#
#                     Sky + RFI peeling: command line interface                 #
#==============================================================================#
#
# Peels the sky sources and then the near-field RFI emitters from MS files
# with the newer solver (zest_sky_rfi_ms!: exact sky model, geometric UVW,
# RK4 StefCal with a residual-power guard).  bin/ttcal_gpu.jl is the original
# peel/zest/shave/prune interface and is unchanged.
#
# USAGE:
#   julia bin/zest_sky_rfi.jl [options] <sources.json> <ms1> [ms2] ...
#
# EXAMPLES:
#   julia bin/zest_sky_rfi.jl --column=DATA sources.json data.ms
#   julia bin/zest_sky_rfi.jl --rfi=rfi.json --rfi-mode=joint --column=DATA sources.json *.ms
#   julia -t 24 bin/zest_sky_rfi.jl --cpu --reg=0.5,0.5 sources.json data.ms
#
#==============================================================================#

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "TTCalX.jl"))
using .TTCalX
using Printf

function print_help()
    println("""
Sky + RFI peeling with TTCalX (RK4 StefCal, exact sky model, residual-power guard)

USAGE:
  julia bin/zest_sky_rfi.jl [options] <sources.json> <ms1> [ms2] ...

OPTIONS:
  --rfi=FILE          Near-field RFI model (JSON entries with "rfi-frequencies")
  --rfi-mode=MODE     after (separate pass after the sky) | joint [default: after]
  --solver=S          rk4 | damped [default: rk4]
  --maxiter=N         Max StefCal iterations per solve [default: 5]
  --tolerance=T       Convergence tolerance [default: 1e-4]
  --minuvw=M          Min baseline length in wavelengths [default: 5.0]
  --peeliter=P        Number of peeling passes [default: 3]
  --max-gain=G        Reject solutions with |g| above G [default: 100]
  --min-elevation=E   Skip sky sources below E degrees [default: 0]
  --reg=O,T[,L]       Tukey phase prior: max offset O deg, taper T deg, weight L [default: off]
  --smooth-flux=S     Tie per-channel gains to a band gain for directions fainter than ~S Jy,
                      so they cannot absorb other sources [default: 0 = off; suggested 1000];
                      --rfi-smooth-flux and --smooth-exempt-flux only matter when it is on
  --rfi-smooth-flux=S The same for the RFI emitters [default: --smooth-flux]
  --smooth-exempt-flux=F  Sky sources with catalog flux >= F Jy keep per-channel gains
                      [default: none exempt; suggested 1e4 (Cas A, Cyg A)]
  --restore=FILE      Point sources (JSON, e.g. Jupiter at its ephemeris position) given back
                      after peeling: what peeling took from them is fitted and added back
  --restore-below-mhz=F  Restore only when the band starts below F MHz [default: always]
  --restore-offset=D  Search radius (deg) for the restored source's position on Stokes V
                      [default: 0; 0.5 for Jupiter]
  --precision=P       double | mixed (normal equations in single precision; ~3-4x faster on
                      GPUs with slow FP64; tolerance floored at 1e-6) [default: double]
  --warm-start        Start each MS from the previous MS's gains when both have the same channels
                      and are <= 60 s apart (consecutive snapshots of one band)
  --peel              Diagonal Jones (peel) instead of full Jones (zest)
  --keep-zeros        Do not flag exactly-zero visibilities
  --cpu               Run on CPU threads (start julia with -t N)
  --column=COL        MS data column to read/write [default: CORRECTED_DATA]
  --verbose           Print every source update (accept / keep_previous / leave_in)
  --quiet             Suppress all output except errors
  --help              Show this help message

EXAMPLES:
  julia bin/zest_sky_rfi.jl --column=DATA sources.json data.ms
  julia bin/zest_sky_rfi.jl --rfi=rfi.json --rfi-mode=joint --column=DATA sources.json *.ms
  julia -t 24 bin/zest_sky_rfi.jl --cpu --reg=0.5,0.5 sources.json data.ms
  julia bin/zest_sky_rfi.jl --rfi=rfi.json --smooth-flux=1000 --smooth-exempt-flux=1e4 \\
        --restore=jupiter.json --restore-below-mhz=41 --restore-offset=0.5 sources.json *.ms
""")
end

function parse_args(args)
    opts = Dict{String,Any}(
        "rfi" => "", "rfi_mode" => :after, "solver" => :rk4, "maxiter" => 5,
        "tolerance" => 1e-4, "minuvw" => 5.0, "peeliter" => 3, "max_gain" => 100.0,
        "min_elevation" => 0.0, "reg" => nothing, "diagonal" => false,
        "flag_zeros" => true, "gpu" => true, "column" => "CORRECTED_DATA",
        "verbosity" => :normal,
        "smooth_flux" => 0.0, "rfi_smooth_flux" => nothing, "smooth_exempt_flux" => Inf,
        "restore" => "", "restore_below_mhz" => Inf, "restore_offset" => 0.0,
        "precision" => :double, "warm_start" => false,
    )
    value(arg) = String(split(arg, "="; limit=2)[2])
    positional = String[]
    for arg in args
        if arg == "--help" || arg == "-h"
            print_help(); exit(0)
        elseif arg == "--quiet";      opts["verbosity"] = :quiet
        elseif arg == "--verbose";    opts["verbosity"] = :verbose
        elseif arg == "--peel";       opts["diagonal"] = true
        elseif arg == "--keep-zeros"; opts["flag_zeros"] = false
        elseif arg == "--cpu";        opts["gpu"] = false
        elseif startswith(arg, "--rfi=");           opts["rfi"] = value(arg)
        elseif startswith(arg, "--rfi-mode=");      opts["rfi_mode"] = Symbol(value(arg))
        elseif startswith(arg, "--solver=");        opts["solver"] = Symbol(value(arg))
        elseif startswith(arg, "--maxiter=");       opts["maxiter"] = parse(Int, value(arg))
        elseif startswith(arg, "--tolerance=");     opts["tolerance"] = parse(Float64, value(arg))
        elseif startswith(arg, "--minuvw=");        opts["minuvw"] = parse(Float64, value(arg))
        elseif startswith(arg, "--peeliter=");      opts["peeliter"] = parse(Int, value(arg))
        elseif startswith(arg, "--max-gain=");      opts["max_gain"] = parse(Float64, value(arg))
        elseif startswith(arg, "--min-elevation="); opts["min_elevation"] = parse(Float64, value(arg))
        elseif startswith(arg, "--reg=")
            opts["reg"] = Tuple(parse.(Float64, split(value(arg), ",")))
            length(opts["reg"]) in (2, 3) || (println("--reg needs O,T or O,T,L"); exit(1))
        elseif startswith(arg, "--column=");        opts["column"] = value(arg)
        elseif arg == "--warm-start";  opts["warm_start"] = true
        elseif startswith(arg, "--precision=");     opts["precision"] = Symbol(value(arg))
        elseif startswith(arg, "--smooth-flux=");   opts["smooth_flux"] = parse(Float64, value(arg))
        elseif startswith(arg, "--rfi-smooth-flux="); opts["rfi_smooth_flux"] = parse(Float64, value(arg))
        elseif startswith(arg, "--smooth-exempt-flux="); opts["smooth_exempt_flux"] = parse(Float64, value(arg))
        elseif startswith(arg, "--restore=");       opts["restore"] = value(arg)
        elseif startswith(arg, "--restore-below-mhz="); opts["restore_below_mhz"] = parse(Float64, value(arg))
        elseif startswith(arg, "--restore-offset="); opts["restore_offset"] = parse(Float64, value(arg))
        elseif startswith(arg, "--")
            println("Unknown option: $arg"); exit(1)
        else
            push!(positional, arg)
        end
    end
    opts["rfi_mode"] in (:after, :joint) || (println("--rfi-mode must be after or joint"); exit(1))
    opts["solver"] in (:rk4, :damped) || (println("--solver must be rk4 or damped"); exit(1))
    opts["precision"] in (:double, :mixed) || (println("--precision must be double or mixed"); exit(1))
    opts["rfi_smooth_flux"] = something(opts["rfi_smooth_flux"], opts["smooth_flux"])
    min(opts["smooth_flux"], opts["rfi_smooth_flux"], opts["restore_offset"]) >= 0 ||
        (println("--smooth-flux, --rfi-smooth-flux and --restore-offset must be >= 0"); exit(1))
    opts["smooth_exempt_flux"] > 0 || (println("--smooth-exempt-flux must be > 0"); exit(1))
    opts["smooth_flux"] + opts["rfi_smooth_flux"] > 0 && opts["reg"] !== nothing &&
        (println("--smooth-flux cannot be combined with --reg"); exit(1))
    if length(positional) < 2
        println("Error: Need at least <sources.json> <ms_file>")
        println("Use --help for usage information")
        exit(1)
    end
    opts["sources"] = positional[1]
    opts["ms_files"] = positional[2:end]
    opts
end

function main()
    (isempty(ARGS) || ARGS[1] in ("--help", "-h")) && (print_help(); exit(0))
    opts = parse_args(ARGS)
    set_verbosity(opts["verbosity"])

    log_header("TTCalX: SKY + RFI PEELING")
    log_section("Configuration")
    log_config(
        "Sources" => opts["sources"],
        "RFI" => isempty(opts["rfi"]) ? "none" : "$(opts["rfi"]) ($(opts["rfi_mode"]))",
        "Jones" => opts["diagonal"] ? "diagonal (peel)" : "full (zest)",
        "Solver" => "$(opts["solver"]), maxiter $(opts["maxiter"]), tol $(opts["tolerance"])",
        "Min UVW" => "$(opts["minuvw"]) λ",
        "Peel iter" => opts["peeliter"],
        "Regularisation" => opts["reg"] === nothing ? "off" : string(opts["reg"]),
        "Smooth flux" => opts["smooth_flux"] + opts["rfi_smooth_flux"] == 0 ? "off" :
            "sky $(opts["smooth_flux"]) Jy, RFI $(opts["rfi_smooth_flux"]) Jy, exempt " *
            (isinf(opts["smooth_exempt_flux"]) ? "none" : "≥ $(opts["smooth_exempt_flux"]) Jy"),
        "Restore" => isempty(opts["restore"]) ? "none" :
            "$(opts["restore"]) (below $(opts["restore_below_mhz"]) MHz, offset ≤ $(opts["restore_offset"])°)",
        "Device" => opts["gpu"] ? "GPU" : "CPU ($(Threads.nthreads()) threads)",
        "Precision" => "$(opts["precision"])$(opts["warm_start"] ? ", warm start" : "")",
        "Column" => opts["column"],
        "MS files" => length(opts["ms_files"]),
    )

    init_pycasacore() || (log_error("python-casacore not available"); exit(1))
    sky = read_gpu_sources(opts["sources"])
    rfi = isempty(opts["rfi"]) ? GPUSource[] : read_gpu_sources(opts["rfi"])
    restore = isempty(opts["restore"]) ? GPUSource[] : read_gpu_sources(opts["restore"])
    for s in restore            # the JSON reader turns these names into a placeholder at (0, 0)
        get_name(s) in ("Sun", "Moon", "Jupiter") &&
            (log_error("--restore: name \"$(get_name(s))\" is read as a placeholder; rename it (e.g. \"$(get_name(s)) (ephemeris)\")"); exit(1))
    end
    log_substep("Loaded $(length(sky)) sky sources, $(length(rfi)) RFI sources, $(length(restore)) to restore")

    report = opts["verbosity"] == :verbose ?
        (name, it; action, p_with, p_new, kw...) ->
            log_detail(@sprintf("%-10s pass %d  %-13s  power with %.4g  new %.4g", name, it, action, p_with, p_new)) :
        nothing

    ms_files = opts["ms_files"]
    gains = nothing
    for (i, ms) in enumerate(ms_files)
        log_section("[$i/$(length(ms_files))] $(basename(ms))")
        t = @elapsed r = zest_sky_rfi_ms!(ms, sky, rfi;
            column=opts["column"], gpu=opts["gpu"], rfi_mode=opts["rfi_mode"],
            solver=opts["solver"], maxiter=opts["maxiter"], tolerance=opts["tolerance"],
            minuvw=opts["minuvw"], peeliter=opts["peeliter"], max_gain=opts["max_gain"],
            min_elevation=opts["min_elevation"], diagonal=opts["diagonal"], reg=opts["reg"],
            flag_zeros=opts["flag_zeros"], report=report,
            smooth_flux=opts["smooth_flux"], rfi_smooth_flux=opts["rfi_smooth_flux"],
            smooth_exempt_flux=opts["smooth_exempt_flux"], restore=restore,
            restore_below_mhz=opts["restore_below_mhz"], restore_offset_deg=opts["restore_offset"],
            precision=opts["precision"], initial_gains=gains)
        gains = opts["warm_start"] ? r.gains : nothing
        log_substep("Peeled: $(isempty(r.peeled) ? "none" : join(r.peeled, ", "))")
        isempty(r.skipped) || log_substep("Below min elevation: $(join(r.skipped, ", "))")
        r.flagged_zeros > 0 && log_substep("Flagged $(r.flagged_zeros) zero samples")
        log_success(@sprintf("Completed in %.2f seconds%s", t, i == 1 ? " (includes JIT)" : ""))
    end
end

main()
