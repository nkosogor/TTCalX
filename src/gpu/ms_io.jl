# Faster Measurement Set I/O and MS helpers.
#
# read_ms_to_gpu / write_gpu_to_ms! convert whole DATA and FLAG columns
# between numpy and Julia through PyCall and gather/scatter rows in Julia
# loops (the write also re-reads both columns and goes back through
# complex128).  read_ms_fast / write_ms_fast! do the row gather/scatter in
# numpy on the arrays casacore returns and give the same results.

"""Mean TIME of an MS as MJD (days, UTC)."""
function ms_mjd(ms_path::String)
    tables == PyNULL() && init_pycasacore()
    t = tables.table(ms_path, readonly=true, ack=false)
    tm = t.getcol("TIME")
    t.close()
    (minimum(tm) + maximum(tm)) / 2 / 86400
end

const _ms_fast_io = PyNULL()

# Returns None whenever the fast path does not apply (several rows per
# baseline, or not 4 polarizations); the callers then use the original code.
const _MS_FAST_IO_PY = """
import numpy as np
from casacore.tables import table


def _cross_rows(t):
    a1 = t.getcol("ANTENNA1")
    a2 = t.getcol("ANTENNA2")
    cross = np.nonzero(a1 != a2)[0]
    lo = np.minimum(a1[cross], a2[cross])
    hi = np.maximum(a1[cross], a2[cross])
    unique = len(np.unique(lo.astype(np.int64) * 1000003 + hi)) == len(cross)
    return a1, a2, cross, lo, hi, unique


def read(ms_path, column):
    t = table(ms_path, ack=False)
    try:
        a1, _, cross, lo, hi, unique = _cross_rows(t)
        if not unique:
            return None
        data = t.getcol(column)
        if data.ndim != 3 or data.shape[2] != 4:
            return None
        data = data[cross]
        flag = t.getcol("FLAG")[cross].any(axis=2)
        uvw = t.getcol("UVW")[cross]
        f = np.asfortranarray
        return (f(lo + 1), f(hi + 1), f(uvw.T.astype(np.float64)),
                f(data[:, :, 0].astype(np.complex128)), f(data[:, :, 1].astype(np.complex128)),
                f(data[:, :, 2].astype(np.complex128)), f(data[:, :, 3].astype(np.complex128)),
                f(flag), len(a1))
    finally:
        t.close()


def write(ms_path, column, index, xx, xy, yx, yy, flags):
    t = table(ms_path, readonly=False, ack=False)
    try:
        a1, a2, _, _, _, unique = _cross_rows(t)
        if not unique:
            return False
        n = max(int(a1.max()), int(a2.max())) + 1
        if n > index.shape[0]:      # antennas without cross baselines in the map
            index = np.pad(index, ((0, n - index.shape[0]), (0, n - index.shape[1])))
        b = index[np.minimum(a1, a2), np.maximum(a1, a2)] - 1
        b[a1 == a2] = -1
        rows = np.nonzero(b >= 0)[0]
        b = b[rows]
        data = t.getcol(column)
        if data.ndim != 3 or data.shape[2] != 4:
            return False
        flag = t.getcol("FLAG")
        for p, v in enumerate((xx, xy, yx, yy)):
            data[rows, :, p] = v[b, :]
            flag[rows, :, p] = flags[b, :]
        t.putcol(column, data)
        t.putcol("FLAG", flag)
        t.flush()
        return True
    finally:
        t.close()
"""

function _fast_io()
    if _ms_fast_io == PyNULL()
        tables == PyNULL() && init_pycasacore()
        mod = pyimport("types").ModuleType("ttcalx_ms_fast_io")
        pybuiltin("exec")(_MS_FAST_IO_PY, mod."__dict__")
        copy!(_ms_fast_io, mod)
    end
    _ms_fast_io
end

"""Channels, antenna positions (3, Nant) and phase centre, parsed as in `read_ms_to_gpu`."""
function _ms_subtables(ms_path::String)
    spw = tables.table(ms_path * "/SPECTRAL_WINDOW", readonly=true)
    chan_freq = spw.getcol("CHAN_FREQ")
    spw.close()
    channels = ndims(chan_freq) == 1 ? Vector{Float64}(chan_freq) :
               size(chan_freq, 1) == 1 ? vec(Float64.(chan_freq[1, :])) : vec(Float64.(chan_freq[:, 1]))
    ant = tables.table(ms_path * "/ANTENNA", readonly=true)
    positions = ant.getcol("POSITION")
    ant.close()
    antenna_positions = ndims(positions) == 2 ?
        (size(positions, 1) == 3 ? Float64.(positions) : Float64.(permutedims(positions))) :
        reshape(Float64.(positions), 3, length(positions) ÷ 3)
    field = tables.table(ms_path * "/FIELD", readonly=true)
    pd = field.getcol("PHASE_DIR")
    field.close()
    ra, dec = if ndims(pd) == 3
        Float64(pd[1, 1, 1]), Float64(pd[1, 1, 2])
    elseif ndims(pd) == 2
        size(pd, 2) == 2 ? (Float64(pd[1, 1]), Float64(pd[1, 2])) : (Float64(pd[1, 1]), Float64(pd[2, 1]))
    elseif ndims(pd) == 1 && length(pd) >= 2
        Float64(pd[1]), Float64(pd[2])
    else
        0.0, π / 2
    end
    channels, antenna_positions, ra, dec
end

"""
    read_ms_fast(ms_path; gpu=true, column="DATA") -> (vis, cal, meta, baseline_dict, Nrows)

Same result as `read_ms_to_gpu`: baselines in order of first appearance as
(min, max) antenna pairs, autocorrelations dropped, flags OR'd over the
polarizations, UVW of the baseline's row, identity calibration.  The row
gather runs in numpy.  MSes with several rows per baseline or other than 4
polarizations fall back to `read_ms_to_gpu`.
"""
function read_ms_fast(ms_path::String; gpu::Bool=true, column::String="DATA")
    r = _fast_io().read(ms_path, column)
    r === nothing && return read_ms_to_gpu(ms_path; gpu=gpu, column=column)
    lo, hi = Vector{Int}(r[1]), Vector{Int}(r[2])
    uvw = Matrix{Float64}(r[3])
    xx, xy, yx, yy = (Matrix{ComplexF64}(x) for x in r[4:7])
    flags, Nrows = Matrix{Bool}(r[8]), Int(r[9])
    channels, antenna_positions, ra, dec = _ms_subtables(ms_path)
    Nbase, Nf = size(xx)
    length(channels) == Nf || return read_ms_to_gpu(ms_path; gpu=gpu, column=column)
    Na = size(antenna_positions, 2)
    baselines = Int32.(permutedims(hcat(lo, hi)))
    baseline_dict = Dict{Tuple{Int,Int}, Int}((lo[α], hi[α]) => α for α in 1:Nbase)
    up = gpu && CUDA.functional() ? CuArray : identity
    vis = GPUVisibilities(up(xx), up(xy), up(yx), up(yy), up(flags))
    meta = GPUMetadata(up(antenna_positions), up(baselines), up(channels), up([0.0, 0.0, 1.0]),
                       ra, dec, up(uvw))
    cal = GPUCalibration(up(ones(ComplexF64, Na, Nf)), up(zeros(ComplexF64, Na, Nf)),
                         up(zeros(ComplexF64, Na, Nf)), up(ones(ComplexF64, Na, Nf)),
                         up(falses(Na, Nf)), true)
    vis, cal, meta, baseline_dict, Nrows
end

"""
    write_ms_fast!(ms_path, vis, baseline_dict, Nrows; column="DATA")

Same result as `write_gpu_to_ms!` (cross-correlation `column` replaced, FLAG
set to the per-baseline flag in all four polarizations, autocorrelations
untouched), with the row scatter in numpy.  Falls back to `write_gpu_to_ms!`
when the fast path does not apply.
"""
function write_ms_fast!(ms_path::String, vis::GPUVisibilities, baseline_dict::Dict, Nrows::Int;
                        column::String="DATA")
    Na = maximum(maximum(k) for k in keys(baseline_dict))
    index = zeros(Int, Na, Na)        # [a1, a2] (1-based antennas) → 1-based baseline index
    for ((a1, a2), α) in baseline_dict
        index[a1, a2] = α
    end
    # PyReverseDims(permutedims(A)) hands numpy a C-ordered view with A's indexing.
    np_view(A) = PyReverseDims(permutedims(A))
    ok = _fast_io().write(ms_path, column, np_view(index),
                          np_view(Array(vis.xx)), np_view(Array(vis.xy)),
                          np_view(Array(vis.yx)), np_view(Array(vis.yy)), np_view(Array(vis.flags)))
    ok || write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column=column)
    nothing
end
