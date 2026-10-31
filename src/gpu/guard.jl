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

# The corrupted model goes into `buf` (one scratch per guarded_zest! call, so the
# loop allocates no visibility-sized arrays).
function _corrupted!(buf, coherency, cal, meta)
    for f in (:xx, :xy, :yx, :yy, :flags)
        copyto!(getfield(buf, f), getfield(coherency, f))
    end
    gpu_corrupt!(buf, cal, meta)
end

# GPU: one fused kernel vis ∓= J₁ C J₂ᴴ (same arithmetic as gpu_corrupt! + gpu_subsrc!/putsrc!,
# without the scratch copy, the intermediate write and read, and the host syncs)
@inline function _corrupt_full(c, g1, g2)
    jv_xx = g1[1] * c[1] + g1[2] * c[3]
    jv_xy = g1[1] * c[2] + g1[2] * c[4]
    jv_yx = g1[3] * c[1] + g1[4] * c[3]
    jv_yy = g1[3] * c[2] + g1[4] * c[4]
    (jv_xx * conj(g2[1]) + jv_xy * conj(g2[2]), jv_xx * conj(g2[3]) + jv_xy * conj(g2[4]),
     jv_yx * conj(g2[1]) + jv_yy * conj(g2[2]), jv_yx * conj(g2[3]) + jv_yy * conj(g2[4]))
end

@inline _corrupt_diag(c, g1, g2) = (g1[1] * c[1] * conj(g2[1]), g1[1] * c[2] * conj(g2[4]),
                                    g1[4] * c[3] * conj(g2[1]), g1[4] * c[4] * conj(g2[4]))

function corrupt_apply_kernel!(dxx, dxy, dyx, dyy, cxx, cxy, cyx, cyy, gxx, gxy, gyx, gyy,
                               baselines, Nbase, Nfreq, Nfreq_cal, diag::Bool, subtract::Bool)
    idx = thread_index_1d()
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        a1 = baselines[1, α]; a2 = baselines[2, α]
        b = Nfreq_cal == 1 ? 1 : β
        c = (cxx[α, β], cxy[α, β], cyx[α, β], cyy[α, β])
        g1 = (gxx[a1, b], gxy[a1, b], gyx[a1, b], gyy[a1, b])
        g2 = (gxx[a2, b], gxy[a2, b], gyx[a2, b], gyy[a2, b])
        m = diag ? _corrupt_diag(c, g1, g2) : _corrupt_full(c, g1, g2)
        if subtract
            dxx[α, β] -= m[1]; dxy[α, β] -= m[2]; dyx[α, β] -= m[3]; dyy[α, β] -= m[4]
        else
            dxx[α, β] += m[1]; dxy[α, β] += m[2]; dyx[α, β] += m[3]; dyy[α, β] += m[4]
        end
    end
    return nothing
end

function _apply_corrupted!(vis, coherency, cal, meta, subtract::Bool)
    Nb, Nf = size(vis.xx)
    blocks, threads = kernel_config_1d(Nb * Nf)
    @cuda blocks=blocks threads=threads corrupt_apply_kernel!(
        vis.xx, vis.xy, vis.yx, vis.yy, coherency.xx, coherency.xy, coherency.yx, coherency.yy,
        cal.xx, cal.xy, cal.yx, cal.yy, meta.baselines, Nb, Nf, Nfreq(cal), cal.is_diagonal, subtract)
    vis
end

const FUSED_CORRUPT = Ref(true)      # false: gpu_corrupt! into a scratch, then subsrc/putsrc
_fused(vis, cal) = FUSED_CORRUPT[] && _is_gpu(vis) && _is_gpu(cal)
_subtract!(vis, coherency, cal, meta, buf) = _fused(vis, cal) ?
    _apply_corrupted!(vis, coherency, cal, meta, true) :
    gpu_subsrc!(vis, _corrupted!(buf, coherency, cal, meta))
_add!(vis, coherency, cal, meta, buf) = _fused(vis, cal) ?
    _apply_corrupted!(vis, coherency, cal, meta, false) :
    gpu_putsrc!(vis, _corrupted!(buf, coherency, cal, meta))
# allocating forms (tests, external callers)
_scratch(vis, coherency) = FUSED_CORRUPT[] && _is_gpu(vis) ? nothing : deepcopy_gpu(coherency)
_subtract!(vis, coherency, cal, meta) = _subtract!(vis, coherency, cal, meta, _scratch(vis, coherency))
_add!(vis, coherency, cal, meta) = _add!(vis, coherency, cal, meta, _scratch(vis, coherency))

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
    all(c -> size(c.xx) == size(vis.xx), coherencies) ||
        throw(DimensionMismatch("every coherency must have the visibilities' size"))
    buf = N == 0 || _fused(vis, calibrations[1]) ? nothing : deepcopy_gpu(coherencies[1])
    length(calibrations) == N || throw(ArgumentError("one calibration per coherency"))
    subtracted = trues(N)
    for s in 1:N
        _subtract!(vis, coherencies[s], calibrations[s], meta, buf)
    end
    for iteration in 1:peeliter, s in 1:N
        coherency, cal = coherencies[s], calibrations[s]
        p_old = subtracted[s] ? residual_power(vis) : Inf
        subtracted[s] && _add!(vis, coherency, cal, meta, buf)
        p_with = residual_power(vis)
        saved = _save_gains(cal)
        converged, niters = solve(cal, vis, coherency)
        finite, gmax = max_gain_amplitude(cal)
        p_new = Inf
        if finite && gmax <= max_gain && (converged || !require_convergence)
            _subtract!(vis, coherency, cal, meta, buf)
            p_new = residual_power(vis)
            if p_new < p_with && p_new <= p_old
                subtracted[s] = true
                report === nothing || report(s, iteration; converged, niters, finite,
                                             max_gain=gmax, p_with, p_old, p_new,
                                             action=:accept)
                continue
            end
            _add!(vis, coherency, cal, meta, buf)       # undo the new subtraction
        end
        _restore_gains!(cal, saved)
        if p_old < p_with
            _subtract!(vis, coherency, cal, meta, buf)
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
