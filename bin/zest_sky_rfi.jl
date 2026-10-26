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
""")
end

function parse_args(args)
    opts = Dict{String,Any}(
        "rfi" => "", "rfi_mode" => :after, "solver" => :rk4, "maxiter" => 5,
        "tolerance" => 1e-4, "minuvw" => 5.0, "peeliter" => 3, "max_gain" => 100.0,
        "min_elevation" => 0.0, "reg" => nothing, "diagonal" => false,
        "flag_zeros" => true, "gpu" => true, "column" => "CORRECTED_DATA",
        "verbosity" => :normal,
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
        elseif startswith(arg, "--")
            println("Unknown option: $arg"); exit(1)
        else
            push!(positional, arg)
        end
    end
    opts["rfi_mode"] in (:after, :joint) || (println("--rfi-mode must be after or joint"); exit(1))
    opts["solver"] in (:rk4, :damped) || (println("--solver must be rk4 or damped"); exit(1))
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
        "Device" => opts["gpu"] ? "GPU" : "CPU ($(Threads.nthreads()) threads)",
        "Column" => opts["column"],
        "MS files" => length(opts["ms_files"]),
    )

    init_pycasacore() || (log_error("python-casacore not available"); exit(1))
    sky = read_gpu_sources(opts["sources"])
    rfi = isempty(opts["rfi"]) ? GPUSource[] : read_gpu_sources(opts["rfi"])
    log_substep("Loaded $(length(sky)) sky sources, $(length(rfi)) RFI sources")

    report = opts["verbosity"] == :verbose ?
        (name, it; action, p_with, p_new, kw...) ->
            log_detail(@sprintf("%-10s pass %d  %-13s  power with %.4g  new %.4g", name, it, action, p_with, p_new)) :
        nothing

    ms_files = opts["ms_files"]
    for (i, ms) in enumerate(ms_files)
        log_section("[$i/$(length(ms_files))] $(basename(ms))")
        t = @elapsed r = zest_sky_rfi_ms!(ms, sky, rfi;
            column=opts["column"], gpu=opts["gpu"], rfi_mode=opts["rfi_mode"],
            solver=opts["solver"], maxiter=opts["maxiter"], tolerance=opts["tolerance"],
            minuvw=opts["minuvw"], peeliter=opts["peeliter"], max_gain=opts["max_gain"],
            min_elevation=opts["min_elevation"], diagonal=opts["diagonal"], reg=opts["reg"],
            flag_zeros=opts["flag_zeros"], report=report)
        log_substep("Peeled: $(isempty(r.peeled) ? "none" : join(r.peeled, ", "))")
        isempty(r.skipped) || log_substep("Below min elevation: $(join(r.skipped, ", "))")
        r.flagged_zeros > 0 && log_substep("Flagged $(r.flagged_zeros) zero samples")
        log_success(@sprintf("Completed in %.2f seconds%s", t, i == 1 ? " (includes JIT)" : ""))
    end
end

main()
