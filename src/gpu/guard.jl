#==============================================================================#
#                     Divergence guards for zest / peel                        #
#==============================================================================#
#
# With few undamped RK4 StefCal iterations (CPU TTCal's zest: maxiter 5,
# minuvw 5) the per-direction solve can diverge for a direction whose model
# fits the data poorly, e.g. Cyg A setting at 0-3 deg elevation (constant
# beam) or the near-field RFI source, and for unflagged all-zero baselines.
# One diverged source (gains ~10-1000) is then subtracted, the next source
# solves against the corrupted residual (Cas A gains blow up in turn), and
# the gains carry over the peel iterations.  On OVRO-LWA 2026-10-24 10h this
# raised the snapshot V noise by 10^2-10^3 on ~10-15% of the 10-s snapshots.
#
# The guards here make each source update a descent step for the residual:
#   flag_zero_visibilities!  flag (baseline, channel) samples that are exactly
#                            zero in XX and YY (dropped data, not sky)
#   guarded_zest!            per-source peel loop that accepts a new solution
#                            only when subtracting it lowers the residual power
#                            (vs. leaving the source in and vs. the previous
#                            gains); otherwise the previous gains are kept, or
#                            the source is left unsubtracted.

"""
    residual_power(vis) -> Float64

Σ|XX|² + Σ|YY|² over the unflagged samples (GPU or host arrays).
"""
residual_power(vis::GPUVisibilities) =
    Float64(sum(abs2, vis.xx .* .!vis.flags) + sum(abs2, vis.yy .* .!vis.flags))

"""
    flag_zero_visibilities!(vis) -> n

Flag every (baseline, channel) sample whose XX and YY are both exactly zero
and return how many unflagged samples were newly flagged.  Exact zeros are
dropped data (correlator/packet loss), not sky: left unflagged, they enter
the gain solve as measurements of zero and can make it diverge.
"""
function flag_zero_visibilities!(vis::GPUVisibilities)
    zero = (vis.xx .== 0) .& (vis.yy .== 0)
    n = Int(sum(zero .& .!vis.flags))
    vis.flags .|= zero
    n
end

_save_gains(cal) = (copy(cal.xx), copy(cal.xy), copy(cal.yx), copy(cal.yy), copy(cal.flags))

function _restore_gains!(cal, saved)
    cal.xx .= saved[1]; cal.xy .= saved[2]; cal.yx .= saved[3]; cal.yy .= saved[4]
    cal.flags .= saved[5]
    cal
end

"""
    max_gain_amplitude(cal) -> (finite, max |g|)
"""
function max_gain_amplitude(cal::GPUCalibration)
    finite, m = true, 0.0
    for values in (cal.xx, cal.xy, cal.yx, cal.yy)
        host = Array(values)
        finite &= all(isfinite, host)
        m = max(m, maximum(abs, host))
    end
    finite, m
end

function _subtract!(vis, coherency, cal, meta)
    c = deepcopy_gpu(coherency)
    gpu_corrupt!(c, cal, meta)
    gpu_subsrc!(vis, c)
end

function _add!(vis, coherency, cal, meta)
    c = deepcopy_gpu(coherency)
    gpu_corrupt!(c, cal, meta)
    gpu_putsrc!(vis, c)
end

"""
    guarded_zest!(vis, meta, coherencies, calibrations, solve;
                  peeliter=3, max_gain=Inf, require_convergence=false,
                  report=nothing) -> subtracted::BitVector

Peel/zest loop of CPU TTCal (`peel!`: subtract every source with its
starting gains, then `peeliter` passes of add back → solve → subtract), with
a residual-power guard on every source update.

`solve(calibration, vis, coherency)` updates `calibration` in place from the
data with the source added back and returns `(converged, niters)`.

A new solution is kept only if its gains are finite, `max|g| ≤ max_gain`,
converged when `require_convergence`, and subtracting it leaves less power
than both leaving the source in (`p_with`) and the previous state (`p_old`,
the source subtracted with its previous gains).  Otherwise the previous gains
are restored and the source is subtracted with them if that beats leaving it
in, or left in the data (the residual never grows from a source update).

`report(index, iteration; converged, niters, finite, max_gain, p_with, p_old,
p_new, action)` is called after every update (`action` is :accept,
:keep_previous or :leave_in).  Returns which sources end up subtracted;
`calibrations` hold the gains that were used.
"""
function guarded_zest!(vis::GPUVisibilities, meta::GPUMetadata,
                       coherencies::AbstractVector, calibrations::AbstractVector, solve;
                       peeliter::Integer=3, max_gain::Real=Inf,
                       require_convergence::Bool=false, report=nothing)
    N = length(coherencies)
    length(calibrations) == N || throw(ArgumentError("one calibration per coherency"))
    subtracted = trues(N)
    for s in 1:N
        _subtract!(vis, coherencies[s], calibrations[s], meta)
    end
    for iteration in 1:peeliter, s in 1:N
        coherency, cal = coherencies[s], calibrations[s]
        p_old = subtracted[s] ? residual_power(vis) : Inf
        subtracted[s] && _add!(vis, coherency, cal, meta)
        p_with = residual_power(vis)
        saved = _save_gains(cal)
        converged, niters = solve(cal, vis, coherency)
        finite, gmax = max_gain_amplitude(cal)
        p_new = Inf
        if finite && gmax <= max_gain && (converged || !require_convergence)
            _subtract!(vis, coherency, cal, meta)
            p_new = residual_power(vis)
            if p_new < p_with && p_new <= p_old
                subtracted[s] = true
                report === nothing || report(s, iteration; converged, niters, finite,
                                             max_gain=gmax, p_with, p_old, p_new,
                                             action=:accept)
                continue
            end
            _add!(vis, coherency, cal, meta)            # undo the new subtraction
        end
        _restore_gains!(cal, saved)
        if p_old < p_with
            _subtract!(vis, coherency, cal, meta)
            subtracted[s] = true
            action = :keep_previous
        else
            subtracted[s] = false
            action = :leave_in
        end
        report === nothing || report(s, iteration; converged, niters, finite,
                                     max_gain=gmax, p_with, p_old, p_new, action)
    end
    subtracted
end
