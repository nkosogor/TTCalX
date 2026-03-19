# MS Bridge — shared processing + PyCall reader/writer
#
# Provides:
#   _assemble_gpu_data()     shared processing (baseline mapping, GPU transfer)
#   _build_write_arrays()    shared write-array construction
#   read_ms_to_gpu()         PyCall-based MS reader (original)
#   write_gpu_to_ms!()       PyCall-based MS writer
#   read_ms_native()         Subprocess-based MS reader (faster, no PyCall overhead)
#   write_ms_native!()       Subprocess-based MS writer

using PyCall

# Import casacore.tables and numpy (cached for performance)
const tables = PyNULL()
const np = PyNULL()

function init_pycasacore()
    try
        copy!(tables, pyimport("casacore.tables"))
        copy!(np, pyimport("numpy"))  # Pre-load numpy
        log_success("python-casacore loaded")
        return true
    catch e
        log_error("Could not load python-casacore: $e")
        println("Install with: pip install python-casacore")
        return false
    end
end

#==============================================================================#
#            Shared processing (used by both PyCall and native readers)         #
#==============================================================================#

"""
    _assemble_gpu_data(raw_data, raw_flags, ant1, ant2, raw_uvw,
                       chan_freq, positions, phase_dir; gpu=true)

Common processing: baseline mapping, visibility extraction, GPU transfer.
Accepts raw arrays from any reader backend (PyCall or native subprocess).
Returns: (vis, cal, meta, baseline_dict, Nrows, row_to_baseline, data_shape)
"""
function _assemble_gpu_data(raw_data, raw_flags, ant1, ant2, raw_uvw,
                            chan_freq, positions, phase_dir; gpu::Bool=true)
    Nrows = length(ant1)

    # --- Parse channels ---
    if ndims(chan_freq) == 1
        Nfreq = length(chan_freq)
        channels = Vector{Float64}(chan_freq)
    elseif size(chan_freq, 1) == 1
        Nfreq = size(chan_freq, 2)
        channels = vec(Float64.(chan_freq[1, :]))
    else
        Nfreq = size(chan_freq, 1)
        channels = vec(Float64.(chan_freq[:, 1]))
    end

    # --- Parse antenna positions (3, Nant) ---
    if ndims(positions) == 2
        if size(positions, 1) == 3
            Nant = size(positions, 2)
            antenna_positions = Float64.(positions)
        else
            Nant = size(positions, 1)
            antenna_positions = Float64.(permutedims(positions))
        end
    else
        Nant = length(positions) ÷ 3
        antenna_positions = reshape(Float64.(positions), 3, Nant)
    end

    # --- Parse phase center ---
    if ndims(phase_dir) == 3
        phase_center_ra = Float64(phase_dir[1, 1, 1])
        phase_center_dec = Float64(phase_dir[1, 1, 2])
    elseif ndims(phase_dir) == 2
        if size(phase_dir, 2) == 2
            phase_center_ra = Float64(phase_dir[1, 1])
            phase_center_dec = Float64(phase_dir[1, 2])
        else
            phase_center_ra = Float64(phase_dir[1, 1])
            phase_center_dec = Float64(phase_dir[2, 1])
        end
    elseif ndims(phase_dir) == 1 && length(phase_dir) >= 2
        phase_center_ra = Float64(phase_dir[1])
        phase_center_dec = Float64(phase_dir[2])
    else
        phase_center_ra = 0.0
        phase_center_dec = π/2
    end

    # --- UVW → (3, Nrows) ---
    if ndims(raw_uvw) == 2
        if size(raw_uvw, 2) == 3
            uvw_all = Float64.(permutedims(raw_uvw))
        else
            uvw_all = Float64.(raw_uvw)
        end
    else
        uvw_all = reshape(Float64.(raw_uvw), 3, Nrows)
    end

    # --- Baseline mapping ---
    baseline_dict = Dict{Tuple{Int,Int}, Int}()
    row_to_baseline = zeros(Int, Nrows)
    baseline_row = Dict{Int, Int}()

    for i in 1:Nrows
        a1 = Int(ant1[i]) + 1
        a2 = Int(ant2[i]) + 1
        if a1 != a2
            key = (min(a1, a2), max(a1, a2))
            if !haskey(baseline_dict, key)
                baseline_dict[key] = length(baseline_dict) + 1
                baseline_row[baseline_dict[key]] = i
            end
            row_to_baseline[i] = baseline_dict[key]
        end
    end

    Nbase = length(baseline_dict)

    baselines = zeros(Int32, 2, Nbase)
    uvw = zeros(Float64, 3, Nbase)
    for ((a1, a2), idx) in baseline_dict
        baselines[1, idx] = a1
        baselines[2, idx] = a2
        row = baseline_row[idx]
        uvw[:, idx] = uvw_all[:, row]
    end

    # --- Determine Nfreq from data shape ---
    data_shape = size(raw_data)
    if length(data_shape) == 3
        if data_shape[3] == 4 || data_shape[3] == 2
            Nfreq_data = data_shape[2]
        elseif data_shape[1] == 4 || data_shape[1] == 2
            Nfreq_data = data_shape[2]
        else
            Nfreq_data = Nfreq
        end
    else
        Nfreq_data = Nfreq
    end

    if Nfreq_data != Nfreq
        Nfreq = Nfreq_data
        if length(channels) != Nfreq
            channels = collect(range(1.0, Float64(Nfreq), length=Nfreq))
        end
    end

    # --- Extract visibilities ---
    vis_xx = zeros(ComplexF64, Nbase, Nfreq)
    vis_xy = zeros(ComplexF64, Nbase, Nfreq)
    vis_yx = zeros(ComplexF64, Nbase, Nfreq)
    vis_yy = zeros(ComplexF64, Nbase, Nfreq)
    vis_flags = zeros(Bool, Nbase, Nfreq)

    if length(data_shape) == 3
        if data_shape[1] == 4 || data_shape[1] == 2
            @inbounds for i in 1:Nrows
                α = row_to_baseline[i]
                if α > 0
                    if data_shape[1] == 4
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[1, β, i]
                            vis_xy[α, β] = raw_data[2, β, i]
                            vis_yx[α, β] = raw_data[3, β, i]
                            vis_yy[α, β] = raw_data[4, β, i]
                            vis_flags[α, β] = raw_flags[1, β, i] | raw_flags[2, β, i] | raw_flags[3, β, i] | raw_flags[4, β, i]
                        end
                    else
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[1, β, i]
                            vis_yy[α, β] = raw_data[2, β, i]
                            vis_flags[α, β] = raw_flags[1, β, i] | raw_flags[2, β, i]
                        end
                    end
                end
            end
        elseif data_shape[3] == 4 || data_shape[3] == 2
            @inbounds for i in 1:Nrows
                α = row_to_baseline[i]
                if α > 0
                    if data_shape[3] == 4
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[i, β, 1]
                            vis_xy[α, β] = raw_data[i, β, 2]
                            vis_yx[α, β] = raw_data[i, β, 3]
                            vis_yy[α, β] = raw_data[i, β, 4]
                            vis_flags[α, β] = raw_flags[i, β, 1] | raw_flags[i, β, 2] | raw_flags[i, β, 3] | raw_flags[i, β, 4]
                        end
                    else
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[i, β, 1]
                            vis_yy[α, β] = raw_data[i, β, 2]
                            vis_flags[α, β] = raw_flags[i, β, 1] | raw_flags[i, β, 2]
                        end
                    end
                end
            end
        end
    end

    # --- GPU transfer ---
    phase_center_lmn = [0.0, 0.0, 1.0]

    if gpu && CUDA.functional()
        vis = GPUVisibilities(
            CuArray(vis_xx), CuArray(vis_xy),
            CuArray(vis_yx), CuArray(vis_yy),
            CuArray(vis_flags)
        )
        meta = GPUMetadata(
            CuArray(antenna_positions), CuArray(baselines),
            CuArray(channels), CuArray(phase_center_lmn),
            phase_center_ra, phase_center_dec, CuArray(uvw)
        )
        cal = GPUCalibration(
            CuArray(ones(ComplexF64, Nant, Nfreq)),
            CuArray(zeros(ComplexF64, Nant, Nfreq)),
            CuArray(zeros(ComplexF64, Nant, Nfreq)),
            CuArray(ones(ComplexF64, Nant, Nfreq)),
            CuArray(falses(Nant, Nfreq)), true
        )
    else
        vis = GPUVisibilities(vis_xx, vis_xy, vis_yx, vis_yy, vis_flags)
        meta = GPUMetadata(antenna_positions, baselines, channels, phase_center_lmn,
                           phase_center_ra, phase_center_dec, uvw)
        cal = GPUCalibration(
            ones(ComplexF64, Nant, Nfreq), zeros(ComplexF64, Nant, Nfreq),
            zeros(ComplexF64, Nant, Nfreq), ones(ComplexF64, Nant, Nfreq),
            falses(Nant, Nfreq), true
        )
    end

    return vis, cal, meta, baseline_dict, Nrows, row_to_baseline, data_shape
end

"""Build output data+flags arrays for MS writing (shared by PyCall and native writers)."""
function _build_write_arrays(vis::GPUVisibilities, row_to_baseline::Vector{Int},
                             data_shape::Tuple)
    xx = Array(vis.xx)
    xy = Array(vis.xy)
    yx = Array(vis.yx)
    yy = Array(vis.yy)
    flags = Array(vis.flags)
    Nbase, Nfreq = size(xx)
    Nrows = length(row_to_baseline)

    new_data = zeros(ComplexF64, data_shape)
    new_flags = ones(Bool, data_shape)

    if data_shape[1] == 4 || data_shape[1] == 2
        @inbounds for i in 1:Nrows
            α = row_to_baseline[i]
            if α > 0
                if data_shape[1] == 4
                    @simd for β in 1:Nfreq
                        new_data[1, β, i] = xx[α, β]
                        new_data[2, β, i] = xy[α, β]
                        new_data[3, β, i] = yx[α, β]
                        new_data[4, β, i] = yy[α, β]
                        f = flags[α, β]
                        new_flags[1, β, i] = f
                        new_flags[2, β, i] = f
                        new_flags[3, β, i] = f
                        new_flags[4, β, i] = f
                    end
                else
                    @simd for β in 1:Nfreq
                        new_data[1, β, i] = xx[α, β]
                        new_data[2, β, i] = yy[α, β]
                        new_flags[1, β, i] = flags[α, β]
                        new_flags[2, β, i] = flags[α, β]
                    end
                end
            end
        end
    elseif data_shape[3] == 4 || data_shape[3] == 2
        @inbounds for i in 1:Nrows
            α = row_to_baseline[i]
            if α > 0
                if data_shape[3] == 4
                    @simd for β in 1:Nfreq
                        new_data[i, β, 1] = xx[α, β]
                        new_data[i, β, 2] = xy[α, β]
                        new_data[i, β, 3] = yx[α, β]
                        new_data[i, β, 4] = yy[α, β]
                        f = flags[α, β]
                        new_flags[i, β, 1] = f
                        new_flags[i, β, 2] = f
                        new_flags[i, β, 3] = f
                        new_flags[i, β, 4] = f
                    end
                else
                    @simd for β in 1:Nfreq
                        new_data[i, β, 1] = xx[α, β]
                        new_data[i, β, 2] = yy[α, β]
                        new_flags[i, β, 1] = flags[α, β]
                        new_flags[i, β, 2] = flags[α, β]
                    end
                end
            end
        end
    end

    return new_data, new_flags
end

#==============================================================================#
#                       PyCall reader / writer                                  #
#==============================================================================#

"""
    read_ms_to_gpu(ms_path; gpu=true, column="DATA")

Read MS using python-casacore via PyCall.
"""
function read_ms_to_gpu(ms_path::String; gpu::Bool=true, column::String="DATA")
    if tables == PyNULL()
        init_pycasacore() || error("python-casacore not available")
    end

    ms = tables.table(ms_path, readonly=true)
    raw_data  = ms.getcol(column)
    raw_flags = ms.getcol("FLAG")
    ant1      = ms.getcol("ANTENNA1")
    ant2      = ms.getcol("ANTENNA2")
    raw_uvw   = ms.getcol("UVW")

    spw = tables.table(ms_path * "/SPECTRAL_WINDOW", readonly=true)
    chan_freq = spw.getcol("CHAN_FREQ")
    spw.close()

    ant_table = tables.table(ms_path * "/ANTENNA", readonly=true)
    positions = ant_table.getcol("POSITION")
    ant_table.close()

    field_table = tables.table(ms_path * "/FIELD", readonly=true)
    phase_dir = field_table.getcol("PHASE_DIR")
    field_table.close()

    ms.close()

    return _assemble_gpu_data(raw_data, raw_flags, ant1, ant2, raw_uvw,
                              chan_freq, positions, phase_dir; gpu=gpu)
end

"""
    write_gpu_to_ms!(ms_path, vis, row_to_baseline, data_shape; column="DATA")

Write visibilities back to MS using python-casacore via PyCall.
"""
function write_gpu_to_ms!(ms_path::String, vis::GPUVisibilities,
                          row_to_baseline::Vector{Int}, data_shape::Tuple;
                          column::String="DATA")
    if tables == PyNULL()
        init_pycasacore() || error("python-casacore not available")
    end

    new_data, new_flags = _build_write_arrays(vis, row_to_baseline, data_shape)

    ms = tables.table(ms_path, readonly=false)
    ms.putcol(column, np.array(new_data, dtype=np.complex128))
    ms.putcol("FLAG", np.array(new_flags, dtype=np.bool_))
    ms.flush()
    ms.close()
end

#==============================================================================#
#                  Native reader / writer (subprocess + binary pipes)           #
#==============================================================================#

const MS_IO_HELPER = joinpath(@__DIR__, "..", "..", "bin", "ms_io_helper.py")

"""Read a self-describing binary array from an IO stream."""
function _read_bin_array(io::IO)
    ndim = read(io, Int32)
    shape = ntuple(_ -> read(io, Int64), ndim)
    dtype_code = read(io, Int32)
    T = (ComplexF64, Float64, Int32, UInt8)[dtype_code + 1]
    n = prod(shape)
    raw = Vector{UInt8}(undef, n * sizeof(T))
    readbytes!(io, raw, length(raw))
    arr = reinterpret(T, raw)
    return reshape(copy(arr), shape...)
end

"""Write a self-describing binary array to an IO stream."""
function _write_bin_array(io::IO, arr::AbstractArray)
    write(io, Int32(ndims(arr)))
    for s in size(arr)
        write(io, Int64(s))
    end
    if eltype(arr) == ComplexF64
        write(io, Int32(0))
    elseif eltype(arr) == Float64
        write(io, Int32(1))
    elseif eltype(arr) == Int32
        write(io, Int32(2))
    else  # UInt8 / Bool
        write(io, Int32(3))
        arr = eltype(arr) == Bool ? UInt8.(arr) : arr
    end
    write(io, Array(arr))
end

"""
    read_ms_native(ms_path; gpu=true, column="DATA")

Read MS using Python subprocess + binary pipe (no PyCall overhead).
"""
function read_ms_native(ms_path::String; gpu::Bool=true, column::String="DATA")
    cmd = `python3 $MS_IO_HELPER read $ms_path $column`
    raw_bytes = read(cmd)
    io = IOBuffer(raw_bytes)

    raw_data  = _read_bin_array(io)
    raw_flags_u8 = _read_bin_array(io)
    ant1      = _read_bin_array(io)
    ant2      = _read_bin_array(io)
    raw_uvw   = _read_bin_array(io)
    chan_freq  = _read_bin_array(io)
    positions = _read_bin_array(io)
    phase_dir = _read_bin_array(io)
    close(io)

    # Convert UInt8 flags to Bool
    raw_flags = raw_flags_u8 .!= 0x00

    return _assemble_gpu_data(raw_data, raw_flags, ant1, ant2, raw_uvw,
                              chan_freq, positions, phase_dir; gpu=gpu)
end

"""
    write_ms_native!(ms_path, vis, row_to_baseline, data_shape; column="DATA")

Write visibilities back to MS using Python subprocess + binary pipe.
"""
function write_ms_native!(ms_path::String, vis::GPUVisibilities,
                          row_to_baseline::Vector{Int}, data_shape::Tuple;
                          column::String="DATA")
    new_data, new_flags = _build_write_arrays(vis, row_to_baseline, data_shape)

    cmd = `python3 $MS_IO_HELPER write $ms_path $column`
    io = open(cmd, "w")
    _write_bin_array(io, new_data)
    _write_bin_array(io, UInt8.(new_flags))
    close(io)
end

"""
    launch_read_async(ms_path; gpu=true, column="DATA", reader=:native)

Launch MS reading in background. Returns a Task that yields the read result.
Only works with native reader (subprocess runs independently of Julia GIL).
Falls back to synchronous read with pycall reader.
"""
function launch_read_async(ms_path::String; gpu::Bool=true,
                           column::String="DATA", reader::Symbol=:native)
    if reader == :native
        return @async read_ms_native(ms_path; gpu=gpu, column=column)
    else
        # PyCall can't run async (GIL), so just wrap in a task for consistent API
        return @async read_ms_to_gpu(ms_path; gpu=gpu, column=column)
    end
end
