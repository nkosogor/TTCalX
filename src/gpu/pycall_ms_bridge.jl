# MS Bridge using python-casacore via PyCall
#
# This provides MS file I/O for GPU-TTCal using python-casacore,
# which is available in most radio astronomy environments.
#
# Requirements:
#   - PyCall.jl 
#   - python-casacore (pip install python-casacore)

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
#                         MS Reading Functions                                  #
#==============================================================================#

"""
    read_ms_to_gpu(ms_path::String; gpu::Bool=true, column::String="DATA")

Read a Measurement Set into GPU-friendly data structures using python-casacore.

Returns: (vis::GPUVisibilities, meta::GPUMetadata)
"""
function read_ms_to_gpu(ms_path::String; gpu::Bool=true, column::String="DATA")
    if tables == PyNULL()
        init_pycasacore() || error("python-casacore not available")
    end
    
    log_detail("Opening MS: $ms_path")
    ms = tables.table(ms_path, readonly=true)
    
    # Read main table data
    log_detail("Reading $column column...")
    raw_data = ms.getcol(column)  # Shape: (Npol, Nfreq, Nrows)
    raw_flags = ms.getcol("FLAG")
    ant1 = ms.getcol("ANTENNA1")
    ant2 = ms.getcol("ANTENNA2")
    raw_uvw = ms.getcol("UVW")  # Shape: (Nrows, 3) or (3, Nrows)
    
    Nrows = length(ant1)
    log_debug("Rows: $Nrows")
    log_debug("UVW shape: $(size(raw_uvw))")
    
    # Read spectral window info
    spw = tables.table(ms_path * "/SPECTRAL_WINDOW", readonly=true)
    chan_freq = spw.getcol("CHAN_FREQ")
    log_debug("CHAN_FREQ shape: $(size(chan_freq))")
    # chan_freq can be (Nfreq,) for single SPW or (Nspw, Nfreq) for multiple
    # PyCall transposes arrays, so check dimensions carefully
    if ndims(chan_freq) == 1
        Nfreq = length(chan_freq)
        channels = Vector{Float64}(chan_freq)
    elseif size(chan_freq, 1) == 1
        # Single SPW, shape is (1, Nfreq)
        Nfreq = size(chan_freq, 2)
        channels = vec(Float64.(chan_freq[1, :]))
    else
        # Multiple SPWs or (Nfreq, 1)
        Nfreq = size(chan_freq, 1)
        channels = vec(Float64.(chan_freq[:, 1]))
    end
    spw.close()
    log_debug("Frequencies: $Nfreq")
    
    # Read antenna positions
    ant_table = tables.table(ms_path * "/ANTENNA", readonly=true)
    positions = ant_table.getcol("POSITION")
    log_debug("Positions shape: $(size(positions))")
    
    # Handle different possible layouts from casacore
    # Typically (3, Nant) but could be (Nant, 3)
    if ndims(positions) == 2
        if size(positions, 1) == 3
            Nant = size(positions, 2)
            antenna_positions = Float64.(positions)
        else
            Nant = size(positions, 1)
            antenna_positions = Float64.(permutedims(positions))
        end
    else
        # 1D array - reshape
        Nant = length(positions) ÷ 3
        antenna_positions = reshape(Float64.(positions), 3, Nant)
    end
    ant_table.close()
    log_debug("Antennas: $Nant")
    
    # Read phase center from FIELD table
    field_table = tables.table(ms_path * "/FIELD", readonly=true)
    phase_dir = field_table.getcol("PHASE_DIR")
    field_table.close()
    log_debug("PHASE_DIR shape: $(size(phase_dir))")
    
    # Extract RA/Dec from phase_dir
    # Shape from python-casacore is typically (Nfield, 1, 2) or (1, 1, 2)
    # where the last dimension is [RA, Dec]
    if ndims(phase_dir) == 3
        # (Nfield, Ndir, 2) - RA/Dec in last dimension
        phase_center_ra = Float64(phase_dir[1, 1, 1])
        phase_center_dec = Float64(phase_dir[1, 1, 2])
    elseif ndims(phase_dir) == 2
        # (Nfield, 2) or (2, Nfield)
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
        log_warning("Could not parse PHASE_DIR, using zenith")
        phase_center_ra = 0.0
        phase_center_dec = π/2  # Zenith
    end
    log_debug("Phase center: RA=$(rad2deg(phase_center_ra))°, Dec=$(rad2deg(phase_center_dec))°")
    
    ms.close()
    
    # Handle UVW shape - can be (Nrows, 3) or (3, Nrows)
    if ndims(raw_uvw) == 2
        if size(raw_uvw, 2) == 3
            # (Nrows, 3) - needs transpose
            uvw_all = Float64.(permutedims(raw_uvw))  # → (3, Nrows)
        else
            # (3, Nrows) - already correct
            uvw_all = Float64.(raw_uvw)
        end
    else
        # 1D - reshape
        uvw_all = reshape(Float64.(raw_uvw), 3, Nrows)
    end
    
    # Build baseline mapping (excluding auto-correlations)
    # Map each unique (ant1, ant2) pair to a baseline index
    baseline_dict = Dict{Tuple{Int,Int}, Int}()
    row_to_baseline = zeros(Int, Nrows)
    baseline_row = Dict{Int, Int}()  # Store first row for each baseline (to get UVW)
    
    for i in 1:Nrows
        a1 = Int(ant1[i]) + 1  # Python 0-indexed → Julia 1-indexed
        a2 = Int(ant2[i]) + 1
        if a1 != a2  # Skip auto-correlations
            key = (min(a1, a2), max(a1, a2))
            if !haskey(baseline_dict, key)
                baseline_dict[key] = length(baseline_dict) + 1
                baseline_row[baseline_dict[key]] = i  # Store first row for UVW
            end
            row_to_baseline[i] = baseline_dict[key]
        end
    end
    
    Nbase = length(baseline_dict)
    log_debug("Baselines: $Nbase (excluding autos)")
    
    # Build baselines array and extract UVW
    baselines = zeros(Int32, 2, Nbase)
    uvw = zeros(Float64, 3, Nbase)
    for ((a1, a2), idx) in baseline_dict
        baselines[1, idx] = a1
        baselines[2, idx] = a2
        row = baseline_row[idx]
        uvw[:, idx] = uvw_all[:, row]
    end
    log_debug("UVW extracted for $Nbase baselines")
    # Extract visibilities into SoA format
    # raw_data shape from casacore via PyCall: typically (Nrows, Nfreq, Npol)
    log_debug("Raw data shape: $(size(raw_data))")
    data_shape = size(raw_data)
    
    # Determine Nfreq from actual data (more reliable than SPECTRAL_WINDOW)
    # Shape is usually (Nrows, Nfreq, Npol)
    if length(data_shape) == 3
        if data_shape[3] == 4 || data_shape[3] == 2
            # (Nrows, Nfreq, Npol)
            Nfreq_data = data_shape[2]
        elseif data_shape[1] == 4 || data_shape[1] == 2
            # (Npol, Nfreq, Nrows)
            Nfreq_data = data_shape[2]
        else
            Nfreq_data = Nfreq
        end
    else
        Nfreq_data = Nfreq
    end
    
    if Nfreq_data != Nfreq
        log_debug("Note: Using Nfreq=$Nfreq_data from DATA (SPECTRAL_WINDOW said $Nfreq)")
        Nfreq = Nfreq_data
        # Regenerate channels if needed
        if length(channels) != Nfreq
            channels = collect(range(1.0, Float64(Nfreq), length=Nfreq))
        end
    end
    
    vis_xx = zeros(ComplexF64, Nbase, Nfreq)
    vis_xy = zeros(ComplexF64, Nbase, Nfreq)
    vis_yx = zeros(ComplexF64, Nbase, Nfreq)
    vis_yy = zeros(ComplexF64, Nbase, Nfreq)
    vis_flags = zeros(Bool, Nbase, Nfreq)
    
    # Handle different possible data layouts - optimized with @inbounds and @simd
    data_shape = size(raw_data)
    if length(data_shape) == 3
        # Determine which axis is which
        if data_shape[1] == 4 || data_shape[1] == 2  # (Npol, Nfreq, Nrows)
            @inbounds for i in 1:Nrows
                α = row_to_baseline[i]
                if α > 0  # Not an auto-correlation
                    if data_shape[1] == 4
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[1, β, i]
                            vis_xy[α, β] = raw_data[2, β, i]
                            vis_yx[α, β] = raw_data[3, β, i]
                            vis_yy[α, β] = raw_data[4, β, i]
                            vis_flags[α, β] = raw_flags[1, β, i] | raw_flags[2, β, i] | raw_flags[3, β, i] | raw_flags[4, β, i]
                        end
                    else  # 2 pols (XX, YY only)
                        @simd for β in 1:Nfreq
                            vis_xx[α, β] = raw_data[1, β, i]
                            vis_yy[α, β] = raw_data[2, β, i]
                            vis_flags[α, β] = raw_flags[1, β, i] | raw_flags[2, β, i]
                        end
                    end
                end
            end
        elseif data_shape[3] == 4 || data_shape[3] == 2  # (Nrows, Nfreq, Npol) - transposed
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
    
    log_debug("Extracted visibilities: $(Nbase) × $(Nfreq)")
    
    # Create GPU structures - use (l,m,n) for legacy compatibility
    phase_center_lmn = [0.0, 0.0, 1.0]  # Zenith direction cosines
    
    if gpu && CUDA.functional()
        log_detail("Transferring to GPU...")
        vis = GPUVisibilities(
            CuArray(vis_xx), CuArray(vis_xy),
            CuArray(vis_yx), CuArray(vis_yy),
            CuArray(vis_flags)
        )
        meta = GPUMetadata(
            CuArray(antenna_positions),
            CuArray(baselines),
            CuArray(channels),
            CuArray(phase_center_lmn),
            phase_center_ra,
            phase_center_dec,
            CuArray(uvw)
        )
    else
        vis = GPUVisibilities(vis_xx, vis_xy, vis_yx, vis_yy, vis_flags)
        meta = GPUMetadata(antenna_positions, baselines, channels, phase_center_lmn, 
                           phase_center_ra, phase_center_dec, uvw)
    end
    
    # Create identity calibration
    cal_xx = ones(ComplexF64, Nant, Nfreq)
    cal_yy = ones(ComplexF64, Nant, Nfreq)
    cal_xy = zeros(ComplexF64, Nant, Nfreq)
    cal_yx = zeros(ComplexF64, Nant, Nfreq)
    
    if gpu && CUDA.functional()
        cal = GPUCalibration(
            CuArray(cal_xx), CuArray(cal_xy),
            CuArray(cal_yx), CuArray(cal_yy),
            CuArray(falses(Nant, Nfreq)), true
        )
    else
        cal = GPUCalibration(cal_xx, cal_xy, cal_yx, cal_yy, falses(Nant, Nfreq), true)
    end
    
    return vis, cal, meta, baseline_dict, Nrows
end


"""
    write_gpu_to_ms!(ms_path::String, vis::GPUVisibilities, 
                     baseline_dict::Dict, Nrows::Int; column::String="DATA")

Write GPU visibilities back to a Measurement Set.
"""
function write_gpu_to_ms!(ms_path::String, vis::GPUVisibilities, 
                          baseline_dict::Dict, Nrows::Int; 
                          column::String="DATA")
    if tables == PyNULL()
        init_pycasacore() || error("python-casacore not available")
    end
    
    log_detail("Opening MS for writing: $ms_path")
    ms = tables.table(ms_path, readonly=false)
    
    # Read antenna columns to rebuild row mapping
    ant1 = ms.getcol("ANTENNA1")
    ant2 = ms.getcol("ANTENNA2")
    
    # Get existing data to know shape
    existing_data = ms.getcol(column)
    data_shape = size(existing_data)
    log_debug("Data shape: $data_shape")
    
    # Transfer from GPU to CPU
    xx = Array(vis.xx)
    xy = Array(vis.xy)
    yx = Array(vis.yx)
    yy = Array(vis.yy)
    flags = Array(vis.flags)
    
    Nbase, Nfreq = size(xx)
    
    # Diagnostic: check if visibilities have been modified
    old_power = sum(abs2, existing_data) / length(existing_data)
    new_vis_power = (sum(abs2, xx) + sum(abs2, xy) + sum(abs2, yx) + sum(abs2, yy)) / (4 * Nbase * Nfreq)
    log_debug("Original data mean power: $old_power")
    log_debug("New visibility mean power: $new_vis_power")
    
    # Build reverse lookup
    baseline_lookup = Dict{Tuple{Int,Int}, Int}()
    for ((a1, a2), idx) in baseline_dict
        baseline_lookup[(a1, a2)] = idx
    end
    
    # Pre-compute row-to-baseline mapping for faster write
    row_to_baseline = zeros(Int, Nrows)
    @inbounds for i in 1:Nrows
        a1 = Int(ant1[i]) + 1
        a2 = Int(ant2[i]) + 1
        if a1 != a2
            key = (min(a1, a2), max(a1, a2))
            row_to_baseline[i] = get(baseline_lookup, key, 0)
        end
    end
    
    # Create output data arrays - use similar() to avoid data copy
    new_data = similar(existing_data)
    copyto!(new_data, existing_data)  # Faster than copy() for large arrays
    new_flags = ms.getcol("FLAG")
    
    # Fill in the data - optimized with @inbounds and @simd
    if data_shape[1] == 4 || data_shape[1] == 2  # (Npol, Nfreq, Nrows)
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
    elseif data_shape[3] == 4 || data_shape[3] == 2  # (Nrows, Nfreq, Npol)
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
    
    # Write back - use cached numpy for conversion
    log_detail("Writing $column column...")
    new_data_np = np.array(new_data, dtype=np.complex128)
    new_flags_np = np.array(new_flags, dtype=np.bool_)
    
    ms.putcol(column, new_data_np)
    ms.putcol("FLAG", new_flags_np)
    ms.flush()
    ms.close()
    
    log_debug("MS updated: $ms_path")
end

# Module initialization message (only in verbose mode)
@verbose begin
    println("PyCall MS Bridge loaded")
    println("Functions:")
    println("  read_ms_to_gpu(ms_path) → (vis, cal, meta, baseline_dict, Nrows)")
    println("  write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows)")
end
