#==============================================================================#
#                              Logging & Progress                               #
#==============================================================================#
#
# Provides two output modes:
#   - Default: Clean progress bars and main steps only
#   - Verbose (--verbose): Detailed diagnostic output
#
#==============================================================================#

using Printf

# Global verbosity state
const VERBOSITY = Ref{Symbol}(:normal)  # :quiet, :normal, :verbose

"""
    set_verbosity(level::Symbol)

Set the global verbosity level: :quiet, :normal, or :verbose
"""
function set_verbosity(level::Symbol)
    @assert level in (:quiet, :normal, :verbose) "Invalid verbosity level"
    VERBOSITY[] = level
end

get_verbosity() = VERBOSITY[]
is_quiet() = VERBOSITY[] == :quiet
is_verbose() = VERBOSITY[] == :verbose
is_normal() = VERBOSITY[] == :normal

#==============================================================================#
# Progress bar
#==============================================================================#

mutable struct ProgressBar
    total::Int
    current::Int
    description::String
    width::Int
    start_time::Float64
    show::Bool
end

"""
    ProgressBar(total; description="", width=40)

Create a progress bar for tracking iterations.
"""
function ProgressBar(total::Int; description::String="", width::Int=40)
    show = !is_quiet()
    ProgressBar(total, 0, description, width, time(), show)
end

function update!(pb::ProgressBar, n::Int=1)
    pb.current += n
    pb.show && render(pb)
end

function finish!(pb::ProgressBar)
    pb.current = pb.total
    if pb.show
        render(pb)
        println()  # New line after progress bar
    end
end

function render(pb::ProgressBar)
    pct = pb.current / pb.total
    filled = round(Int, pct * pb.width)
    bar = "█" ^ filled * "░" ^ (pb.width - filled)
    
    elapsed = time() - pb.start_time
    if pb.current > 0 && pb.current < pb.total
        eta = elapsed / pb.current * (pb.total - pb.current)
        eta_str = format_time(eta)
    else
        eta_str = "--:--"
    end
    
    desc = isempty(pb.description) ? "" : "$(pb.description) "
    print("\r$(desc)|$(bar)| $(pb.current)/$(pb.total) [$(format_time(elapsed))<$(eta_str)]")
end

function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.0fs", seconds)
    elseif seconds < 3600
        m, s = divrem(seconds, 60)
        return @sprintf("%dm%02ds", m, s)
    else
        h, rem = divrem(seconds, 3600)
        m, s = divrem(rem, 60)
        return @sprintf("%dh%02dm", h, m)
    end
end

#==============================================================================#
# Logging functions
#==============================================================================#

"""
    log_header(title)

Print a prominent header (always shown unless quiet).
"""
function log_header(title::String)
    is_quiet() && return
    println("═" ^ 70)
    println("  $title")
    println("═" ^ 70)
end

"""
    log_section(title)

Print a section divider (always shown unless quiet).
"""
function log_section(title::String)
    is_quiet() && return
    println("\n─── $title " * "─" ^ max(0, 54 - length(title)))
end

"""
    log_step(msg)

Print a main step message (always shown unless quiet).
"""
function log_step(msg::String)
    is_quiet() && return
    println("▶ $msg")
end

"""
    log_substep(msg)

Print a substep message (shown in normal and verbose modes).
"""
function log_substep(msg::String)
    is_quiet() && return
    println("  • $msg")
end

"""
    log_detail(msg)

Print detailed diagnostic info (only in verbose mode).
"""
function log_detail(msg::String)
    is_verbose() || return
    println("    $msg")
end

"""
    log_debug(msg)

Print debug-level info (only in verbose mode).
"""
function log_debug(msg::String)
    is_verbose() || return
    println("      ⋮ $msg")
end

"""
    log_success(msg)

Print a success message.
"""
function log_success(msg::String)
    is_quiet() && return
    println("✓ $msg")
end

"""
    log_warning(msg)

Print a warning message (always shown unless quiet).
"""
function log_warning(msg::String)
    is_quiet() && return
    println("⚠ $msg")
end

"""
    log_error(msg)

Print an error message (always shown).
"""
function log_error(msg::String)
    println("✗ ERROR: $msg")
end

"""
    log_config(pairs...)

Print configuration key-value pairs nicely formatted.
"""
function log_config(pairs::Pair{String,<:Any}...)
    is_quiet() && return
    max_key_len = maximum(length(p.first) for p in pairs)
    for (k, v) in pairs
        println("  $(rpad(k * ":", max_key_len + 1)) $v")
    end
end

"""
    log_table_row(items...; widths=nothing)

Print a formatted table row.
"""
function log_table_row(items...; widths=nothing)
    is_quiet() && return
    if widths === nothing
        println("  " * join(items, "  "))
    else
        parts = [rpad(string(item), w) for (item, w) in zip(items, widths)]
        println("  " * join(parts, "  "))
    end
end

#==============================================================================#
# Convenient macro for verbose-only blocks
#==============================================================================#

"""
    @verbose expr

Execute expression only if verbose mode is enabled.
"""
macro verbose(expr)
    quote
        if is_verbose()
            $(esc(expr))
        end
    end
end

"""
    @normal expr

Execute expression in normal or verbose mode (not quiet).
"""
macro normal(expr)
    quote
        if !is_quiet()
            $(esc(expr))
        end
    end
end
