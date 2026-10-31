# Multithreaded host (CPU) backend for zesting.
#
# TTCalX's peel operations launch CUDA kernels unconditionally, and its cpu_*
# fallbacks are single-threaded (cpu_makesquare! also differs from the GPU
# kernel: it keeps autocorrelations and accumulates).  These methods run the
# GPU kernels' arithmetic on host arrays, one channel per Julia thread, so a
# zest with host `GPUVisibilities` (e.g. `read_ms_to_gpu(ms; gpu=false)`)
# gives the GPU result.  Start Julia with `-t N` (or JULIA_NUM_THREADS).

const HostVisibilities = GPUVisibilities{<:Array}

_channel(a::AbstractMatrix, β) = view(a, :, β:β)
_channel(a::AbstractArray{<:Any,3}, β) = view(a, :, :, β:β)
_channel(v::GPUVisibilities, β) = GPUVisibilities(_channel(v.xx, β), _channel(v.xy, β),
                                                  _channel(v.yx, β), _channel(v.yy, β), _channel(v.flags, β))
_channel(c::GPUCalibration, β) = GPUCalibration(_channel(c.xx, β), _channel(c.xy, β), _channel(c.yx, β),
                                                _channel(c.yy, β), _channel(c.flags, β), c.is_diagonal)

"""Corrupt host model visibilities by the gains, V → J₁ V J₂ᴴ (one channel per thread)."""
function gpu_corrupt!(vis::HostVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nfreq(cal) == 1 && return cpu_corrupt!(vis, cal, meta)
    Threads.@threads for β in 1:Nfreq(vis)
        cpu_corrupt!(_channel(vis, β), _channel(cal, β), meta)
    end
    vis
end

gpu_subsrc!(data::HostVisibilities, model::GPUVisibilities) = cpu_subsrc!(data, model)
gpu_putsrc!(data::HostVisibilities, model::GPUVisibilities) = cpu_putsrc!(data, model)

"""
    host_makesquare!(sq, vis, meta)

Baseline-indexed host visibilities to Hermitian Nant×Nant×Nfreq matrices, as
`makesquare_kernel!`: flagged visibilities and autocorrelations are left at
zero, [a1, a2] = V and [a2, a1] = Vᴴ.
"""
function host_makesquare!(sq::GPUSquareVisibilities, vis::GPUVisibilities, meta::GPUMetadata)
    size(sq.xx) == (meta.Nant, meta.Nant, Nfreq(vis)) || throw(DimensionMismatch(
        "square visibilities are $(size(sq.xx)), expected $((meta.Nant, meta.Nant, Nfreq(vis))) (no wideband on the host)"))
    bl = meta.baselines
    Threads.@threads for β in 1:Nfreq(vis)
        for a in (sq.xx, sq.xy, sq.yx, sq.yy)
            fill!(view(a, :, :, β), zero(eltype(a)))
        end
        @inbounds for α in 1:meta.Nbase
            vis.flags[α, β] && continue
            a1, a2 = bl[1, α], bl[2, α]
            a1 == a2 && continue
            vxx, vxy, vyx, vyy = vis.xx[α, β], vis.xy[α, β], vis.yx[α, β], vis.yy[α, β]
            sq.xx[a1, a2, β] = vxx; sq.xy[a1, a2, β] = vxy
            sq.yx[a1, a2, β] = vyx; sq.yy[a1, a2, β] = vyy
            sq.xx[a2, a1, β] = conj(vxx); sq.xy[a2, a1, β] = conj(vyx)
            sq.yx[a2, a1, β] = conj(vxy); sq.yy[a2, a1, β] = conj(vyy)
        end
    end
    sq
end
