# GPU Peel/Zest Implementation
# Full calibration-based source removal with GPU/CPU acceleration
#
# This implements:
#   - Peeling: diagonal Jones matrices, per frequency channel
#   - Shaving: diagonal Jones matrices, wideband (one per subband)
#   - Zesting: full Jones matrices, per frequency channel
#   - Pruning: full Jones matrices, wideband (one per subband)

using CUDA
using LinearAlgebra
using Printf

# Sources must be included before this file in GPUTTCal.jl

# GPU_C is defined in kernels/utils.jl

#==============================================================================#
#                        CPU/GPU Helpers                                        #
#==============================================================================#

"""Check if an array is on GPU."""
_is_gpu(x::CuArray) = true
_is_gpu(x::AbstractArray) = false
_is_gpu(vis::GPUVisibilities) = _is_gpu(vis.xx)
_is_gpu(cal::GPUCalibration) = _is_gpu(cal.xx)
_is_gpu(meta::GPUMetadata) = _is_gpu(meta.uvw)

"""Create zeros array on GPU or CPU."""
_zeros(::Type{T}, dims...; gpu::Bool=true) where T = 
    gpu && CUDA.functional() ? CUDA.zeros(T, dims...) : zeros(T, dims...)

"""Create ones array on GPU or CPU."""
_ones(::Type{T}, dims...; gpu::Bool=true) where T =
    gpu && CUDA.functional() ? CUDA.ones(T, dims...) : ones(T, dims...)

#==============================================================================#
#                      Minimum UV-W Baseline Filtering                         #
#==============================================================================#

"""
    flag_short_baselines!(sq::GPUSquareVisibilities, meta::GPUMetadata, minuvw::Float64)

Flag short baselines by zeroing out visibility entries where baseline length < minuvw * λ.
This matches the original TTCal behavior of `flag_short_baselines!` which is critical
for good calibration quality (short baselines are dominated by extended emission).
"""
function flag_short_baselines!(sq::GPUSquareVisibilities, meta::GPUMetadata, minuvw::Float64)
    minuvw <= 0 && return sq  # No flagging needed
    
    c = 299792458.0  # Speed of light
    Na = meta.Nant
    Nf = size(sq.xx, 3)
    
    # Get antenna positions on CPU
    positions = meta.antenna_positions isa CuArray ? Array(meta.antenna_positions) : meta.antenna_positions
    channels = meta.channels isa CuArray ? Array(meta.channels) : meta.channels
    
    # Get square visibility arrays on CPU for modification
    sq_xx = sq.xx isa CuArray ? Array(sq.xx) : sq.xx
    sq_xy = sq.xy isa CuArray ? Array(sq.xy) : sq.xy
    sq_yx = sq.yx isa CuArray ? Array(sq.yx) : sq.yx
    sq_yy = sq.yy isa CuArray ? Array(sq.yy) : sq.yy
    
    @inbounds for β in 1:Nf
        # Wavelength for this channel
        λ = c / channels[min(β, length(channels))]  # Handle wideband case
        min_baseline_m = minuvw * λ
        
        for ant1 in 1:Na
            for ant2 in 1:Na
                ant1 == ant2 && continue
                
                # Compute baseline length from antenna positions
                dx = positions[1, ant1] - positions[1, ant2]
                dy = positions[2, ant1] - positions[2, ant2]
                dz = positions[3, ant1] - positions[3, ant2]
                baseline_length = sqrt(dx^2 + dy^2 + dz^2)
                
                if baseline_length < min_baseline_m
                    # Flag by zeroing
                    sq_xx[ant1, ant2, β] = zero(ComplexF64)
                    sq_xy[ant1, ant2, β] = zero(ComplexF64)
                    sq_yx[ant1, ant2, β] = zero(ComplexF64)
                    sq_yy[ant1, ant2, β] = zero(ComplexF64)
                end
            end
        end
    end
    
    # Copy back to GPU if needed
    if sq.xx isa CuArray
        copyto!(sq.xx, CuArray(sq_xx))
        copyto!(sq.xy, CuArray(sq_xy))
        copyto!(sq.yx, CuArray(sq_yx))
        copyto!(sq.yy, CuArray(sq_yy))
    else
        copyto!(sq.xx, sq_xx)
        copyto!(sq.xy, sq_xy)
        copyto!(sq.yx, sq_yx)
        copyto!(sq.yy, sq_yy)
    end
    
    return sq
end

#==============================================================================#
#                        Calibration Solver                                     #
#==============================================================================#

"""
    gpu_stefcal!(calibration, measured, model, meta; maxiter=20, tolerance=1e-3, minuvw=0.0)

Run Stefcal solver on GPU to find calibration gains.

Minimizes ||V - J₁ M J₂'||² where:
- V is measured visibilities
- M is model visibilities (coherencies)
- J₁, J₂ are antenna gains

For diagonal Jones: only solves for xx and yy gains.
For full Jones: solves for all four Jones components.

# Keyword Arguments
- `maxiter::Int=20`: Maximum stefcal iterations
- `tolerance::Float64=1e-3`: Convergence tolerance
- `minuvw::Float64=0.0`: Minimum baseline length in wavelengths (excludes short baselines)
"""
function gpu_stefcal!(calibration::GPUCalibration, 
                      measured::GPUVisibilities, 
                      model::GPUVisibilities, 
                      meta::GPUMetadata;
                      maxiter::Int=20, 
                      tolerance::Float64=1e-3,
                      minuvw::Float64=0.0)
    Na = meta.Nant
    Nf = calibration.is_diagonal ? Nfreq(calibration) : Nfreq(calibration)
    Nb = meta.Nbase
    
    # Detect if we're in GPU or CPU mode
    use_gpu = _is_gpu(measured)
    
    # Check if we're doing wideband or per-channel
    wideband = Nfreq(calibration) == 1 && Nfreq(measured) > 1
    
    # Create square visibility matrices for antenna-antenna indexing
    meas_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(measured), gpu=use_gpu)
    model_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(model), gpu=use_gpu)
    
    # Convert from baseline to square format
    if use_gpu
        gpu_makesquare!(meas_sq, measured, meta)
        gpu_makesquare!(model_sq, model, meta)
    else
        cpu_makesquare!(meas_sq, measured, meta)
        cpu_makesquare!(model_sq, model, meta)
    end
    
    # Flag short baselines if minuvw > 0 (critical for good calibration)
    if minuvw > 0
        flag_short_baselines!(meas_sq, meta, minuvw)
        flag_short_baselines!(model_sq, meta, minuvw)
    end
    
    # Allocate step arrays
    if calibration.is_diagonal
        step_xx = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
        step_yy = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
    else
        step_xx = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
        step_xy = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
        step_yx = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
        step_yy = _zeros(ComplexF64, Na, Nfreq(calibration); gpu=use_gpu)
    end
    
    converged = false
    niters = 0
    
    for iter = 1:maxiter
        niters = iter
        
        # Compute step: step = newgain - oldgain (what stefcal wants to move by)
        if calibration.is_diagonal
            if use_gpu
                gpu_stefcal_step_diagonal!(step_xx, step_yy, calibration, meas_sq, model_sq, meta)
            else
                cpu_stefcal_step_diagonal!(step_xx, step_yy, calibration, meas_sq, model_sq, meta)
            end
            
            # Check convergence BEFORE applying step (matching original TTCal)
            # Original: vecnorm(δ) < tolerance * vecnorm(x) where δ is step, x is current gains
            norm_step = sqrt(sum(abs2.(step_xx)) + sum(abs2.(step_yy)))
            norm_gains = sqrt(sum(abs2.(calibration.xx)) + sum(abs2.(calibration.yy)))
            
            if norm_step < tolerance * norm_gains
                converged = true
            end
            
            # Apply half step (damping for stability in simple iteration)
            calibration.xx .+= 0.5 .* step_xx
            calibration.yy .+= 0.5 .* step_yy
        else
            if use_gpu
                gpu_stefcal_step_full!(step_xx, step_xy, step_yx, step_yy, calibration, meas_sq, model_sq, meta)
            else
                cpu_stefcal_step_full!(step_xx, step_xy, step_yx, step_yy, calibration, meas_sq, model_sq, meta)
            end
            
            # Check convergence BEFORE applying step
            norm_step = sqrt(sum(abs2.(step_xx)) + sum(abs2.(step_xy)) + 
                            sum(abs2.(step_yx)) + sum(abs2.(step_yy)))
            norm_gains = sqrt(sum(abs2.(calibration.xx)) + sum(abs2.(calibration.xy)) +
                             sum(abs2.(calibration.yx)) + sum(abs2.(calibration.yy)))
            
            if norm_step < tolerance * norm_gains
                converged = true
            end
            
            # Apply half step (damping for stability)
            calibration.xx .+= 0.5 .* step_xx
            calibration.xy .+= 0.5 .* step_xy
            calibration.yx .+= 0.5 .* step_yx
            calibration.yy .+= 0.5 .* step_yy
        end
        
        converged && break
    end
    
    return converged, niters
end

"""
GPU-accelerated makesquare: convert baseline-indexed to antenna-antenna indexed.
"""
function gpu_makesquare!(sq::GPUSquareVisibilities, vis::GPUVisibilities, meta::GPUMetadata)
    Na = meta.Nant
    Nb = meta.Nbase
    Nf = Nfreq(vis)
    Nf_sq = size(sq.xx, 3)
    
    # Zero out the output
    fill!(sq.xx, zero(ComplexF64))
    fill!(sq.xy, zero(ComplexF64))
    fill!(sq.yx, zero(ComplexF64))
    fill!(sq.yy, zero(ComplexF64))
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    if Nf_sq == 1 && Nf > 1
        # Wideband: sum over all frequencies using separate real/imag arrays
        # CUDA doesn't support atomic operations on ComplexF64, so we split
        sq_xx_re = CUDA.zeros(Float64, Na, Na)
        sq_xx_im = CUDA.zeros(Float64, Na, Na)
        sq_xy_re = CUDA.zeros(Float64, Na, Na)
        sq_xy_im = CUDA.zeros(Float64, Na, Na)
        sq_yx_re = CUDA.zeros(Float64, Na, Na)
        sq_yx_im = CUDA.zeros(Float64, Na, Na)
        sq_yy_re = CUDA.zeros(Float64, Na, Na)
        sq_yy_im = CUDA.zeros(Float64, Na, Na)
        
        @cuda blocks=blocks threads=threads makesquare_wideband_kernel!(
            sq_xx_re, sq_xx_im, sq_xy_re, sq_xy_im,
            sq_yx_re, sq_yx_im, sq_yy_re, sq_yy_im,
            vis.xx, vis.xy, vis.yx, vis.yy, vis.flags,
            meta.baselines,
            Na, Nb, Nf
        )
        CUDA.synchronize()
        
        # Combine real and imaginary parts back into complex arrays
        N = Na * Na
        blocks2, threads2 = kernel_config_1d(N)
        @cuda blocks=blocks2 threads=threads2 combine_complex_kernel!(
            view(sq.xx, :, :, 1), sq_xx_re, sq_xx_im, N)
        @cuda blocks=blocks2 threads=threads2 combine_complex_kernel!(
            view(sq.xy, :, :, 1), sq_xy_re, sq_xy_im, N)
        @cuda blocks=blocks2 threads=threads2 combine_complex_kernel!(
            view(sq.yx, :, :, 1), sq_yx_re, sq_yx_im, N)
        @cuda blocks=blocks2 threads=threads2 combine_complex_kernel!(
            view(sq.yy, :, :, 1), sq_yy_re, sq_yy_im, N)
    else
        # Per-channel
        @cuda blocks=blocks threads=threads makesquare_kernel!(
            sq.xx, sq.xy, sq.yx, sq.yy,
            vis.xx, vis.xy, vis.yx, vis.yy, vis.flags,
            meta.baselines,
            Na, Nb, Nf
        )
    end
    
    CUDA.synchronize()
    sq
end

"""
Helper function for atomic complex addition using real-valued atomics.
CUDA doesn't support atomic operations on ComplexF64 directly,
so we use reinterpret to work with the real and imaginary parts separately.
"""
@inline function atomic_add_complex!(arr, i, j, k, val::ComplexF64)
    # Reinterpret the complex array as Float64 to access real/imag parts
    # arr[i,j,k] has real part at linear index and imag at linear index + 1
    # Use pointer arithmetic to add to real and imaginary parts separately
    ptr = pointer(arr, LinearIndices(arr)[i, j, k])
    CUDA.atomic_add!(ptr, real(val))
    CUDA.atomic_add!(ptr + sizeof(Float64), imag(val))
    return nothing
end

"""Wideband makesquare: sum across all frequencies using reinterpreted real atomics."""
function makesquare_wideband_kernel!(
    sq_xx_re, sq_xx_im, sq_xy_re, sq_xy_im, 
    sq_yx_re, sq_yx_im, sq_yy_re, sq_yy_im,
    vis_xx, vis_xy, vis_yx, vis_yy, vis_flags,
    baselines,
    Nant, Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        if vis_flags[α, β]
            return nothing
        end
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        if ant1 == ant2
            return nothing
        end
        
        vxx = vis_xx[α, β]
        vxy = vis_xy[α, β]
        vyx = vis_yx[α, β]
        vyy = vis_yy[α, β]
        
        # Atomic add for wideband using separate real/imag arrays
        CUDA.@atomic sq_xx_re[ant1, ant2] += real(vxx)
        CUDA.@atomic sq_xx_im[ant1, ant2] += imag(vxx)
        CUDA.@atomic sq_xy_re[ant1, ant2] += real(vxy)
        CUDA.@atomic sq_xy_im[ant1, ant2] += imag(vxy)
        CUDA.@atomic sq_yx_re[ant1, ant2] += real(vyx)
        CUDA.@atomic sq_yx_im[ant1, ant2] += imag(vyx)
        CUDA.@atomic sq_yy_re[ant1, ant2] += real(vyy)
        CUDA.@atomic sq_yy_im[ant1, ant2] += imag(vyy)
        
        # Conjugate transpose entries
        CUDA.@atomic sq_xx_re[ant2, ant1] += real(vxx)
        CUDA.@atomic sq_xx_im[ant2, ant1] -= imag(vxx)  # conjugate: -imag
        CUDA.@atomic sq_xy_re[ant2, ant1] += real(vyx)
        CUDA.@atomic sq_xy_im[ant2, ant1] -= imag(vyx)
        CUDA.@atomic sq_yx_re[ant2, ant1] += real(vxy)
        CUDA.@atomic sq_yx_im[ant2, ant1] -= imag(vxy)
        CUDA.@atomic sq_yy_re[ant2, ant1] += real(vyy)
        CUDA.@atomic sq_yy_im[ant2, ant1] -= imag(vyy)
    end
    
    return nothing
end

"""Kernel to combine real and imaginary parts back into complex array."""
function combine_complex_kernel!(sq, sq_re, sq_im, N)
    idx = thread_index_1d()
    if idx <= N
        sq[idx] = complex(sq_re[idx], sq_im[idx])
    end
    return nothing
end

"""Diagonal stefcal step wrapper."""
function gpu_stefcal_step_diagonal!(step_xx, step_yy, cal, meas_sq, model_sq, meta)
    Na = meta.Nant
    Nf = Nfreq(cal)
    
    blocks, threads = kernel_config_1d(Na * Nf)
    
    @cuda blocks=blocks threads=threads stefcal_step_diagonal_kernel!(
        step_xx, step_yy,
        cal.xx, cal.yy,
        meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy,
        model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy,
        Na, Nf
    )
    
    CUDA.synchronize()
end

"""Full Jones stefcal step wrapper."""
function gpu_stefcal_step_full!(step_xx, step_xy, step_yx, step_yy, cal, meas_sq, model_sq, meta)
    Na = meta.Nant
    Nf = Nfreq(cal)
    
    blocks, threads = kernel_config_1d(Na * Nf)
    
    @cuda blocks=blocks threads=threads stefcal_step_full_kernel!(
        step_xx, step_xy, step_yx, step_yy,
        cal.xx, cal.xy, cal.yx, cal.yy,
        meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy,
        model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy,
        Na, Nf
    )
    
    CUDA.synchronize()
end

#==============================================================================#
#                        CPU Implementations                                    #
#==============================================================================#

"""CPU makesquare: convert baseline-indexed to antenna-antenna indexed."""
function cpu_makesquare!(sq::GPUSquareVisibilities, vis::GPUVisibilities, meta::GPUMetadata)
    Na = meta.Nant
    Nb = meta.Nbase
    Nf = Nfreq(vis)
    Nf_sq = size(sq.xx, 3)
    
    # Zero out the output
    fill!(sq.xx, zero(ComplexF64))
    fill!(sq.xy, zero(ComplexF64))
    fill!(sq.yx, zero(ComplexF64))
    fill!(sq.yy, zero(ComplexF64))
    
    wideband = Nf_sq == 1 && Nf > 1
    
    @inbounds for α in 1:Nb
        ant1 = meta.baselines[1, α]
        ant2 = meta.baselines[2, α]
        
        for β in 1:Nf
            if vis.flags[α, β]
                continue
            end
            
            vxx = vis.xx[α, β]
            vxy = vis.xy[α, β]
            vyx = vis.yx[α, β]
            vyy = vis.yy[α, β]
            
            β_out = wideband ? 1 : β
            
            # V[ant1, ant2]
            sq.xx[ant1, ant2, β_out] += vxx
            sq.xy[ant1, ant2, β_out] += vxy
            sq.yx[ant1, ant2, β_out] += vyx
            sq.yy[ant1, ant2, β_out] += vyy
            
            # V[ant2, ant1] = V[ant1, ant2]' (conjugate transpose)
            sq.xx[ant2, ant1, β_out] += conj(vxx)
            sq.xy[ant2, ant1, β_out] += conj(vyx)
            sq.yx[ant2, ant1, β_out] += conj(vxy)
            sq.yy[ant2, ant1, β_out] += conj(vyy)
        end
    end
    
    sq
end

"""CPU diagonal stefcal step - matches GPU kernel exactly."""
function cpu_stefcal_step_diagonal!(step_xx, step_yy, cal, meas_sq, model_sq, meta)
    Na = meta.Nant
    Nf = Nfreq(cal)
    
    fill!(step_xx, zero(ComplexF64))
    fill!(step_yy, zero(ComplexF64))
    
    @inbounds for β in 1:Nf
        for j in 1:Na  # antenna j (solving for)
            # Accumulators for numerator and denominator
            num_xx = zero(ComplexF64)
            num_yy = zero(ComplexF64)
            den_xx = zero(ComplexF64)
            den_yy = zero(ComplexF64)
            
            # Sum over all antennas i
            for i in 1:Na
                # Get current gain for antenna i
                gi_xx = cal.xx[i, β]
                gi_yy = cal.yy[i, β]
                
                # Get model visibility M[i,j]
                m_xx = model_sq.xx[i, j, β]
                m_xy = model_sq.xy[i, j, β]
                m_yx = model_sq.yx[i, j, β]
                m_yy = model_sq.yy[i, j, β]
                
                # Get measured visibility V[i,j]
                v_xx = meas_sq.xx[i, j, β]
                v_xy = meas_sq.xy[i, j, β]
                v_yx = meas_sq.yx[i, j, β]
                v_yy = meas_sq.yy[i, j, β]
                
                # GM = G[i] * M[i,j] where G is diagonal
                gm_xx = gi_xx * m_xx
                gm_xy = gi_xx * m_xy
                gm_yx = gi_yy * m_yx
                gm_yy = gi_yy * m_yy
                
                # numerator += (GM.xx'*V.xx + GM.yx'*V.yx, GM.xy'*V.xy + GM.yy'*V.yy)
                num_xx += conj(gm_xx) * v_xx + conj(gm_yx) * v_yx
                num_yy += conj(gm_xy) * v_xy + conj(gm_yy) * v_yy
                
                # denominator += (GM.xx'*GM.xx + GM.yx'*GM.yx, GM.xy'*GM.xy + GM.yy'*GM.yy)
                den_xx += conj(gm_xx) * gm_xx + conj(gm_yx) * gm_yx
                den_yy += conj(gm_xy) * gm_xy + conj(gm_yy) * gm_yy
            end
            
            # Compute step: step = conj(numerator / denominator) - current
            ok_xx = abs(den_xx) > eps(Float64)
            ok_yy = abs(den_yy) > eps(Float64)
            
            new_xx = ok_xx ? conj(num_xx / den_xx) : cal.xx[j, β]
            new_yy = ok_yy ? conj(num_yy / den_yy) : cal.yy[j, β]
            
            step_xx[j, β] = new_xx - cal.xx[j, β]
            step_yy[j, β] = new_yy - cal.yy[j, β]
        end
    end
end

"""CPU full Jones stefcal step - matches GPU kernel exactly."""
function cpu_stefcal_step_full!(step_xx, step_xy, step_yx, step_yy, cal, meas_sq, model_sq, meta)
    Na = meta.Nant
    Nf = Nfreq(cal)
    
    fill!(step_xx, zero(ComplexF64))
    fill!(step_xy, zero(ComplexF64))
    fill!(step_yx, zero(ComplexF64))
    fill!(step_yy, zero(ComplexF64))
    
    @inbounds for β in 1:Nf
        for j in 1:Na  # antenna j (solving for)
            # Accumulators for 2x2 numerator and denominator matrices
            num_xx = zero(ComplexF64)
            num_xy = zero(ComplexF64)
            num_yx = zero(ComplexF64)
            num_yy = zero(ComplexF64)
            
            den_xx = zero(ComplexF64)
            den_xy = zero(ComplexF64)
            den_yx = zero(ComplexF64)
            den_yy = zero(ComplexF64)
            
            for i in 1:Na
                # Get current gain for antenna i (full Jones)
                gi_xx = cal.xx[i, β]
                gi_xy = cal.xy[i, β]
                gi_yx = cal.yx[i, β]
                gi_yy = cal.yy[i, β]
                
                # Get model visibility M[i,j]
                m_xx = model_sq.xx[i, j, β]
                m_xy = model_sq.xy[i, j, β]
                m_yx = model_sq.yx[i, j, β]
                m_yy = model_sq.yy[i, j, β]
                
                # Get measured visibility V[i,j]
                v_xx = meas_sq.xx[i, j, β]
                v_xy = meas_sq.xy[i, j, β]
                v_yx = meas_sq.yx[i, j, β]
                v_yy = meas_sq.yy[i, j, β]
                
                # GM = G[i] * M[i,j] (full matrix multiply)
                gm_xx = gi_xx*m_xx + gi_xy*m_yx
                gm_xy = gi_xx*m_xy + gi_xy*m_yy
                gm_yx = gi_yx*m_xx + gi_yy*m_yx
                gm_yy = gi_yx*m_xy + gi_yy*m_yy
                
                # numerator: GM' * V
                num_xx += conj(gm_xx)*v_xx + conj(gm_yx)*v_yx
                num_xy += conj(gm_xx)*v_xy + conj(gm_yx)*v_yy
                num_yx += conj(gm_xy)*v_xx + conj(gm_yy)*v_yx
                num_yy += conj(gm_xy)*v_xy + conj(gm_yy)*v_yy
                
                # denominator: GM' * GM
                den_xx += conj(gm_xx)*gm_xx + conj(gm_yx)*gm_yx
                den_xy += conj(gm_xx)*gm_xy + conj(gm_yx)*gm_yy
                den_yx += conj(gm_xy)*gm_xx + conj(gm_yy)*gm_yx
                den_yy += conj(gm_xy)*gm_xy + conj(gm_yy)*gm_yy
            end
            
            # Solve: step = conj_transpose(denominator \ numerator) - current
            det_den = den_xx*den_yy - den_xy*den_yx
            ok = abs(det_den) > eps(Float64)
            
            if ok
                inv_det = 1.0 / det_den
                
                # Inverse of denominator
                inv_xx = den_yy * inv_det
                inv_xy = -den_xy * inv_det
                inv_yx = -den_yx * inv_det
                inv_yy = den_xx * inv_det
                
                # Solve: sol = inv(den) * num
                sol_xx = inv_xx*num_xx + inv_xy*num_yx
                sol_xy = inv_xx*num_xy + inv_xy*num_yy
                sol_yx = inv_yx*num_xx + inv_yy*num_yx
                sol_yy = inv_yx*num_xy + inv_yy*num_yy
                
                # Conjugate transpose of solution
                new_xx = conj(sol_xx)
                new_xy = conj(sol_yx)  # xy and yx swap in transpose
                new_yx = conj(sol_xy)
                new_yy = conj(sol_yy)
                
                step_xx[j, β] = new_xx - cal.xx[j, β]
                step_xy[j, β] = new_xy - cal.xy[j, β]
                step_yx[j, β] = new_yx - cal.yx[j, β]
                step_yy[j, β] = new_yy - cal.yy[j, β]
            end
            # else steps remain zero (already initialized)
        end
    end
end

#==============================================================================#
#                       Model Visibility Generation                             #
#==============================================================================#

"""
    gpu_genvis(meta, source; phase_center_ra, phase_center_dec, lst)

Generate model visibilities for a source on GPU.
"""
function gpu_genvis(meta::GPUMetadata, source::GPUSource;
                    phase_center_ra::Float64=0.0,
                    phase_center_dec::Float64=0.0,
                    lst::Float64=0.0)
    Nb = meta.Nbase
    Nf = meta.Nfreq
    
    vis = GPUVisibilities(Nb, Nf, gpu=true)
    gpu_genvis!(vis, meta, source, phase_center_ra, phase_center_dec, lst)
    return vis
end

"""
    gpu_genvis!(vis, meta, source; ...)

Generate model visibilities and add to existing visibility data.
"""
function gpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUSource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64)
    # Get source direction relative to phase center
    lmn = source_direction_lmn(source, phase_center_ra, phase_center_dec, lst)
    l, m, n = lmn
    
    # Compute flux at each frequency
    Nf = meta.Nfreq
    channels_cpu = Array(meta.channels)
    
    flux_I = zeros(Float64, Nf)
    flux_Q = zeros(Float64, Nf)
    flux_U = zeros(Float64, Nf)
    flux_V = zeros(Float64, Nf)
    
    for β in 1:Nf
        I, Q, U, V = source.spectrum(channels_cpu[β])
        flux_I[β] = I
        flux_Q[β] = Q
        flux_U[β] = U
        flux_V[β] = V
    end
    
    # Convert Stokes to Jones/visibility (assuming linear feeds)
    # V_xx = I + Q, V_yy = I - Q, V_xy = U + iV, V_yx = U - iV
    flux_xx = CuArray(Complex.(flux_I .+ flux_Q))
    flux_xy = CuArray(Complex.(flux_U, flux_V))
    flux_yx = CuArray(Complex.(flux_U, .-flux_V))
    flux_yy = CuArray(Complex.(flux_I .- flux_Q))
    
    Nb = meta.Nbase
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    if source isa GPUGaussianSource
        # Gaussian source - need to compute shape factor
        major_fwhm = source.major_fwhm
        minor_fwhm = source.minor_fwhm
        
        # Position angle correction for sources far from phase center
        # The position angle is defined relative to local north at the source,
        # but (u,v) is defined relative to phase center. We need to rotate
        # the position angle by the angle between source's local north and v-axis.
        Δra = source.ra - phase_center_ra
        sin_dra = sin(Δra)
        cos_dra = cos(Δra)
        sin_dec = sin(source.dec)
        cos_dec = cos(source.dec)
        tan_dec_pc = tan(phase_center_dec)
        
        # Rotation angle from source's local north to v-axis
        rotation_angle = atan(sin_dra, cos_dec * tan_dec_pc - sin_dec * cos_dra)
        
        # Corrected position angle in (u,v) frame
        pa = source.position_angle + rotation_angle
        
        @cuda blocks=blocks threads=threads genvis_gaussian_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            meta.uvw,
            flux_xx, flux_xy, flux_yx, flux_yy,
            l, m, n,
            major_fwhm, minor_fwhm, pa,
            meta.channels,
            Nb, Nf
        )
    else
        # Point source
        @cuda blocks=blocks threads=threads genvis_point_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            meta.uvw,
            flux_xx, flux_xy, flux_yx, flux_yy,
            l, m, n,
            meta.channels,
            Nb, Nf
        )
    end
    
    CUDA.synchronize()
    return vis
end

"""Point source visibility generation kernel."""
function genvis_point_kernel!(
    vis_xx, vis_xy, vis_yx, vis_yy,
    uvw,
    flux_xx, flux_xy, flux_yx, flux_yy,
    l, m, n,
    channels,
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1  # baseline
        β = ((idx - 1) ÷ Nbase) + 1  # frequency
        
        # Get UVW for this baseline
        u = uvw[1, α]
        v = uvw[2, α]
        w = uvw[3, α]
        
        # Wavelength (c / freq)
        λ = 299792458.0 / channels[β]
        
        # Phase: -2π/λ * (u*l + v*m + w*(n-1)) - negated to match CASA/ITRF convention
        phase = -2π * (u * l + v * m + w * (n - 1)) / λ
        
        # Complex exponential (fringe)
        fringe = exp(im * phase)
        
        # Add visibility contribution
        vis_xx[α, β] += flux_xx[β] * fringe
        vis_xy[α, β] += flux_xy[β] * fringe
        vis_yx[α, β] += flux_yx[β] * fringe
        vis_yy[α, β] += flux_yy[β] * fringe
    end
    
    return nothing
end

"""Gaussian source visibility generation kernel."""
function genvis_gaussian_kernel!(
    vis_xx, vis_xy, vis_yx, vis_yy,
    uvw,
    flux_xx, flux_xy, flux_yx, flux_yy,
    l, m, n,
    major_fwhm, minor_fwhm, pa,
    channels,
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1  # baseline
        β = ((idx - 1) ÷ Nbase) + 1  # frequency
        
        # Get UVW for this baseline
        u = uvw[1, α]
        v = uvw[2, α]
        w = uvw[3, α]
        
        # Wavelength (c / freq)
        λ = 299792458.0 / channels[β]
        
        # Phase: -2π/λ * (u*l + v*m + w*(n-1)) - negated to match CASA/ITRF convention
        phase = -2π * (u * l + v * m + w * (n - 1)) / λ
        
        # Complex exponential (fringe)
        fringe = exp(im * phase)
        
        # Gaussian shape factor
        # Rotate UV into source frame
        cos_pa = cos(pa)
        sin_pa = sin(pa)
        u_rot = u * cos_pa + v * sin_pa
        v_rot = -u * sin_pa + v * cos_pa
        
        # FWHM to sigma: sigma = FWHM / (2*sqrt(2*ln(2)))
        # Visibility envelope: exp(-2π² σ² |u|²)
        # = exp(-π² * FWHM² * |u|² / (2*ln(2)))
        sigma_maj = major_fwhm / (2 * sqrt(2 * log(2)))
        sigma_min = minor_fwhm / (2 * sqrt(2 * log(2)))
        
        # UV in wavelengths
        u_lam = u_rot / λ
        v_lam = v_rot / λ
        
        # Gaussian envelope
        envelope = exp(-2π^2 * (sigma_maj^2 * u_lam^2 + sigma_min^2 * v_lam^2))
        
        # Add visibility contribution
        vis_xx[α, β] += flux_xx[β] * fringe * envelope
        vis_xy[α, β] += flux_xy[β] * fringe * envelope
        vis_yx[α, β] += flux_yx[β] * fringe * envelope
        vis_yy[α, β] += flux_yy[β] * fringe * envelope
    end
    
    return nothing
end

"""Handle multi-component sources by iterating."""
function gpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUMultiSource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64)
    for component in source.components
        gpu_genvis!(vis, meta, component, phase_center_ra, phase_center_dec, lst)
    end
    return vis
end

#==============================================================================#
#                            Corrupt/Apply Cal                                  #
#==============================================================================#

"""
    gpu_corrupt!(vis, calibration, meta)

Corrupt model visibilities by calibration gains: V → J₁ V J₂'
"""
function gpu_corrupt!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nb = meta.Nbase
    Nf = meta.Nfreq
    Nf_cal = Nfreq(cal)
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    if cal.is_diagonal
        @cuda blocks=blocks threads=threads corrupt_diagonal_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            cal.xx, cal.yy,
            meta.baselines,
            Nb, Nf, Nf_cal
        )
    else
        @cuda blocks=blocks threads=threads corrupt_full_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            cal.xx, cal.xy, cal.yx, cal.yy,
            meta.baselines,
            Nb, Nf, Nf_cal
        )
    end
    
    CUDA.synchronize()
    vis
end

"""Diagonal corrupt kernel."""
function corrupt_diagonal_kernel!(
    vis_xx, vis_xy, vis_yx, vis_yy,
    cal_xx, cal_yy,
    baselines,
    Nbase, Nfreq, Nfreq_cal
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Get calibration (handle wideband case)
        β_cal = Nfreq_cal == 1 ? 1 : β
        
        j1_xx = cal_xx[ant1, β_cal]
        j1_yy = cal_yy[ant1, β_cal]
        j2_xx = cal_xx[ant2, β_cal]
        j2_yy = cal_yy[ant2, β_cal]
        
        # Get visibilities
        vxx = vis_xx[α, β]
        vxy = vis_xy[α, β]
        vyx = vis_yx[α, β]
        vyy = vis_yy[α, β]
        
        # Apply: V → J₁ V J₂'
        # For diagonal J: Vxx' = j1_xx * Vxx * conj(j2_xx)
        vis_xx[α, β] = j1_xx * vxx * conj(j2_xx)
        vis_xy[α, β] = j1_xx * vxy * conj(j2_yy)
        vis_yx[α, β] = j1_yy * vyx * conj(j2_xx)
        vis_yy[α, β] = j1_yy * vyy * conj(j2_yy)
    end
    
    return nothing
end

"""Full Jones corrupt kernel."""
function corrupt_full_kernel!(
    vis_xx, vis_xy, vis_yx, vis_yy,
    cal_xx, cal_xy, cal_yx, cal_yy,
    baselines,
    Nbase, Nfreq, Nfreq_cal
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        β_cal = Nfreq_cal == 1 ? 1 : β
        
        # J1
        j1_xx = cal_xx[ant1, β_cal]
        j1_xy = cal_xy[ant1, β_cal]
        j1_yx = cal_yx[ant1, β_cal]
        j1_yy = cal_yy[ant1, β_cal]
        
        # J2
        j2_xx = cal_xx[ant2, β_cal]
        j2_xy = cal_xy[ant2, β_cal]
        j2_yx = cal_yx[ant2, β_cal]
        j2_yy = cal_yy[ant2, β_cal]
        
        # V
        vxx = vis_xx[α, β]
        vxy = vis_xy[α, β]
        vyx = vis_yx[α, β]
        vyy = vis_yy[α, β]
        
        # J1 * V
        jv_xx = j1_xx * vxx + j1_xy * vyx
        jv_xy = j1_xx * vxy + j1_xy * vyy
        jv_yx = j1_yx * vxx + j1_yy * vyx
        jv_yy = j1_yx * vxy + j1_yy * vyy
        
        # (J1 * V) * J2' where J2' = [[conj(j2_xx), conj(j2_yx)], [conj(j2_xy), conj(j2_yy)]]
        vis_xx[α, β] = jv_xx * conj(j2_xx) + jv_xy * conj(j2_xy)
        vis_xy[α, β] = jv_xx * conj(j2_yx) + jv_xy * conj(j2_yy)
        vis_yx[α, β] = jv_yx * conj(j2_xx) + jv_yy * conj(j2_xy)
        vis_yy[α, β] = jv_yx * conj(j2_yx) + jv_yy * conj(j2_yy)
    end
    
    return nothing
end

#==============================================================================#
#                        CPU Genvis/Corrupt                                     #
#==============================================================================#

"""CPU version of genvis for a single source."""
function cpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUSource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64)
    # Get source direction relative to phase center
    lmn = source_direction_lmn(source, phase_center_ra, phase_center_dec, lst)
    l, m, n = lmn
    
    # Get channels as regular array
    channels = meta.channels isa CuArray ? Array(meta.channels) : meta.channels
    uvw = meta.uvw isa CuArray ? Array(meta.uvw) : meta.uvw
    
    Nf = meta.Nfreq
    Nb = meta.Nbase
    
    # Compute flux at each frequency
    flux_I = zeros(Float64, Nf)
    flux_Q = zeros(Float64, Nf)
    flux_U = zeros(Float64, Nf)
    flux_V = zeros(Float64, Nf)
    
    for β in 1:Nf
        I, Q, U, V = source.spectrum(channels[β])
        flux_I[β] = I
        flux_Q[β] = Q
        flux_U[β] = U
        flux_V[β] = V
    end
    
    # Convert Stokes to visibility
    flux_xx = Complex.(flux_I .+ flux_Q)
    flux_xy = Complex.(flux_U, flux_V)
    flux_yx = Complex.(flux_U, .-flux_V)
    flux_yy = Complex.(flux_I .- flux_Q)
    
    # Gaussian source parameters
    is_gaussian = source isa GPUGaussianSource
    if is_gaussian
        major_fwhm = source.major_fwhm
        minor_fwhm = source.minor_fwhm
        
        # Position angle correction for sources far from phase center
        # The position angle is defined relative to local north at the source,
        # but (u,v) is defined relative to phase center. We need to rotate
        # the position angle by the angle between source's local north and v-axis.
        Δra = source.ra - phase_center_ra
        sin_dra = sin(Δra)
        cos_dra = cos(Δra)
        sin_dec = sin(source.dec)
        cos_dec = cos(source.dec)
        tan_dec_pc = tan(phase_center_dec)
        
        # Rotation angle from source's local north to v-axis
        rotation_angle = atan(sin_dra, cos_dec * tan_dec_pc - sin_dec * cos_dra)
        
        # Corrected position angle in (u,v) frame
        pa = source.position_angle + rotation_angle
        
        sigma_maj = major_fwhm / (2 * sqrt(2 * log(2)))
        sigma_min = minor_fwhm / (2 * sqrt(2 * log(2)))
        cos_pa = cos(pa)
        sin_pa = sin(pa)
    end
    
    c = 299792458.0
    
    @inbounds for β in 1:Nf
        λ = c / channels[β]
        
        for α in 1:Nb
            u = uvw[1, α]
            v = uvw[2, α]
            w = uvw[3, α]
            
            # Phase - negated to match CASA/ITRF convention
            phase = -2π * (u * l + v * m + w * (n - 1)) / λ
            fringe = exp(im * phase)
            
            # Gaussian envelope
            if is_gaussian
                u_rot = u * cos_pa + v * sin_pa
                v_rot = -u * sin_pa + v * cos_pa
                u_lam = u_rot / λ
                v_lam = v_rot / λ
                envelope = exp(-2π^2 * (sigma_maj^2 * u_lam^2 + sigma_min^2 * v_lam^2))
            else
                envelope = 1.0
            end
            
            # Add visibility
            vis.xx[α, β] += flux_xx[β] * fringe * envelope
            vis.xy[α, β] += flux_xy[β] * fringe * envelope
            vis.yx[α, β] += flux_yx[β] * fringe * envelope
            vis.yy[α, β] += flux_yy[β] * fringe * envelope
        end
    end
    
    return vis
end

"""CPU version of genvis for multi-source."""
function cpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source::GPUMultiSource,
                     phase_center_ra::Float64, phase_center_dec::Float64, lst::Float64)
    for component in source.components
        cpu_genvis!(vis, meta, component, phase_center_ra, phase_center_dec, lst)
    end
    return vis
end

"""CPU version of corrupt."""
function cpu_corrupt!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nb = meta.Nbase
    Nf = size(vis.xx, 2)
    Nf_cal = Nfreq(cal)
    
    baselines = meta.baselines isa CuArray ? Array(meta.baselines) : meta.baselines
    
    @inbounds for β in 1:Nf
        β_cal = Nf_cal == 1 ? 1 : β
        
        for α in 1:Nb
            ant1 = baselines[1, α]
            ant2 = baselines[2, α]
            
            if cal.is_diagonal
                j1_xx = cal.xx[ant1, β_cal]
                j1_yy = cal.yy[ant1, β_cal]
                j2_xx = cal.xx[ant2, β_cal]
                j2_yy = cal.yy[ant2, β_cal]
                
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                vis.xx[α, β] = j1_xx * vxx * conj(j2_xx)
                vis.xy[α, β] = j1_xx * vxy * conj(j2_yy)
                vis.yx[α, β] = j1_yy * vyx * conj(j2_xx)
                vis.yy[α, β] = j1_yy * vyy * conj(j2_yy)
            else
                j1_xx = cal.xx[ant1, β_cal]
                j1_xy = cal.xy[ant1, β_cal]
                j1_yx = cal.yx[ant1, β_cal]
                j1_yy = cal.yy[ant1, β_cal]
                
                j2_xx = cal.xx[ant2, β_cal]
                j2_xy = cal.xy[ant2, β_cal]
                j2_yx = cal.yx[ant2, β_cal]
                j2_yy = cal.yy[ant2, β_cal]
                
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                # J1 * V
                jv_xx = j1_xx * vxx + j1_xy * vyx
                jv_xy = j1_xx * vxy + j1_xy * vyy
                jv_yx = j1_yx * vxx + j1_yy * vyx
                jv_yy = j1_yx * vxy + j1_yy * vyy
                
                # (J1 * V) * J2'
                vis.xx[α, β] = jv_xx * conj(j2_xx) + jv_xy * conj(j2_xy)
                vis.xy[α, β] = jv_xx * conj(j2_yx) + jv_xy * conj(j2_yy)
                vis.yx[α, β] = jv_yx * conj(j2_xx) + jv_yy * conj(j2_xy)
                vis.yy[α, β] = jv_yx * conj(j2_yx) + jv_yy * conj(j2_yy)
            end
        end
    end
    
    return vis
end

"""CPU subtract source."""
function cpu_subsrc!(vis::GPUVisibilities, model::GPUVisibilities)
    vis.xx .-= model.xx
    vis.xy .-= model.xy
    vis.yx .-= model.yx
    vis.yy .-= model.yy
    return vis
end

"""CPU add source back."""
function cpu_putsrc!(vis::GPUVisibilities, model::GPUVisibilities)
    vis.xx .+= model.xx
    vis.xy .+= model.xy
    vis.yx .+= model.yx
    vis.yy .+= model.yy
    return vis
end

#==============================================================================#
#                          Main Peel Functions                                  #
#==============================================================================#

"""
    peel_gpu!(vis, meta, sources; kwargs...)

Peel sources from visibilities using diagonal Jones matrices (one per channel).
Automatically uses GPU or CPU based on array types.

# Arguments
- `vis::GPUVisibilities`: Measured visibilities (modified in place)
- `meta::GPUMetadata`: Interferometer metadata
- `sources::Vector{<:AbstractGPUPeelingSource}`: Sources to peel

# Keyword Arguments
- `peeliter::Int=3`: Number of peeling iterations
- `maxiter::Int=20`: Max iterations per stefcal solve
- `tolerance::Float64=1e-3`: Convergence tolerance
- `minuvw::Float64=0.0`: Minimum baseline length in wavelengths
- `phase_center_ra::Float64=0.0`: Phase center RA (radians)
- `phase_center_dec::Float64=0.0`: Phase center Dec (radians)
- `lst::Float64=0.0`: Local sidereal time (radians)

# Returns
- `calibrations`: Vector of GPUCalibration for each source

# Notes
Output verbosity is controlled by `set_verbosity(:quiet)`, `set_verbosity(:normal)`, 
or `set_verbosity(:verbose)`.
"""
function peel_gpu!(vis::GPUVisibilities, meta::GPUMetadata, 
                   sources::Vector{T};
                   peeliter::Int=3, maxiter::Int=20, tolerance::Float64=1e-3,
                   minuvw::Float64=0.0,
                   phase_center_ra::Float64=0.0, phase_center_dec::Float64=0.0,
                   lst::Float64=0.0) where {T<:AbstractGPUPeelingSource}
    
    Nsources = length(sources)
    Na = meta.Nant
    Nf = meta.Nfreq
    Nb = meta.Nbase
    
    # Detect GPU vs CPU mode based on visibility array type
    use_gpu = _is_gpu(vis)
    mode_str = use_gpu ? "GPU" : "CPU"
    
    log_step("$mode_str Peel: $Nsources sources, $Na antennas, $Nf frequencies")
    log_detail("Min UVW: $minuvw λ")
    
    # Initialize calibrations for each source
    calibrations = [calibration_type(s, Na, Nf, gpu=use_gpu) for s in sources]
    
    # Generate model visibilities (coherencies) for each source
    log_step("Generating model visibilities...")
    coherencies = GPUVisibilities[]
    pb_genvis = ProgressBar(Nsources; description="  Model vis")
    for s in sources
        coh = GPUVisibilities(Nb, Nf, gpu=use_gpu)
        if use_gpu
            gpu_genvis!(coh, meta, unwrap(s), phase_center_ra, phase_center_dec, lst)
        else
            cpu_genvis!(coh, meta, unwrap(s), phase_center_ra, phase_center_dec, lst)
        end
        push!(coherencies, coh)
        update!(pb_genvis)
    end
    finish!(pb_genvis)
    
    # Select appropriate functions based on mode
    corrupt! = use_gpu ? gpu_corrupt! : cpu_corrupt!
    subsrc! = use_gpu ? gpu_subsrc! : cpu_subsrc!
    putsrc! = use_gpu ? gpu_putsrc! : cpu_putsrc!
    
    # Initial subtraction of all sources
    log_step("Subtracting initial source estimates...")
    for (s, coh, cal) in zip(1:Nsources, coherencies, calibrations)
        corrupted = deepcopy_gpu(coh)
        corrupt!(corrupted, cal, meta)
        subsrc!(vis, corrupted)
    end
    
    # Peel iterations
    log_step("Peeling ($peeliter iterations)...")
    for iter in 1:peeliter
        log_substep("Iteration $iter/$peeliter")
        pb_peel = ProgressBar(Nsources; description="    Sources")
        
        for s in 1:Nsources
            coh = coherencies[s]
            cal = calibrations[s]
            
            # Put source back
            corrupted = deepcopy_gpu(coh)
            corrupt!(corrupted, cal, meta)
            putsrc!(vis, corrupted)
            
            # Solve for calibration (stefcal handles GPU/CPU internally)
            converged, niters = gpu_stefcal!(cal, vis, coh, meta, 
                                             maxiter=maxiter, tolerance=tolerance,
                                             minuvw=minuvw)
            
            if !converged
                log_detail("Source $s ($(get_name(sources[s]))): did not converge ($niters iters)")
            end
            
            # Subtract with updated calibration
            corrupted = deepcopy_gpu(coh)
            corrupt!(corrupted, cal, meta)
            subsrc!(vis, corrupted)
            
            update!(pb_peel)
        end
        finish!(pb_peel)
    end
    
    log_success("Peeling complete")
    return calibrations
end

# Convenience methods for legacy-style calls
"""
    peel_gpu!(vis, meta, sources; kwargs...)

Peel with diagonal Jones gains (per channel). Convenience wrapper that
converts `GPUSource` objects into `GPUPeelingSource` and calls `peel_gpu!`.
"""
function peel_gpu!(vis::GPUVisibilities, meta::GPUMetadata, 
                   sources::Vector{<:GPUSource}; kwargs...)
    peeling_sources = [GPUPeelingSource(s) for s in sources]
    peel_gpu!(vis, meta, peeling_sources; kwargs...)
end

"""
    shave_gpu!(vis, meta, sources; kwargs...)

Shave with diagonal Jones gains (wideband). Converts `GPUSource` objects
into `GPUShavingSource` and calls `peel_gpu!`.
"""
function shave_gpu!(vis::GPUVisibilities, meta::GPUMetadata,
                    sources::Vector{<:GPUSource}; kwargs...)
    shaving_sources = [GPUShavingSource(s) for s in sources]
    peel_gpu!(vis, meta, shaving_sources; kwargs...)
end

"""
    zest_gpu!(vis, meta, sources; kwargs...)

Zest with full Jones gains (per channel). Converts `GPUSource` objects
into `GPUZestingSource` and calls `peel_gpu!`.
"""
function zest_gpu!(vis::GPUVisibilities, meta::GPUMetadata,
                   sources::Vector{<:GPUSource}; kwargs...)
    zesting_sources = [GPUZestingSource(s) for s in sources]
    peel_gpu!(vis, meta, zesting_sources; kwargs...)
end

"""
    prune_gpu!(vis, meta, sources; kwargs...)

Prune with full Jones gains (wideband). Converts `GPUSource` objects
into `GPUPruningSource` and calls `peel_gpu!`.
"""
function prune_gpu!(vis::GPUVisibilities, meta::GPUMetadata,
                    sources::Vector{<:GPUSource}; kwargs...)
    pruning_sources = [GPUPruningSource(s) for s in sources]
    peel_gpu!(vis, meta, pruning_sources; kwargs...)
end

#==============================================================================#
#                           Helper Functions                                    #
#==============================================================================#

"""Deep copy for GPU visibilities."""
function deepcopy_gpu(vis::GPUVisibilities)
    GPUVisibilities(
        copy(vis.xx), copy(vis.xy), copy(vis.yx), copy(vis.yy),
        copy(vis.flags)
    )
end

"""Deep copy for GPU calibration."""
function deepcopy_gpu(cal::GPUCalibration)
    GPUCalibration(
        copy(cal.xx), copy(cal.xy), copy(cal.yx), copy(cal.yy),
        copy(cal.flags), cal.is_diagonal
    )
end

# kernel_config_1d and thread_index_1d are defined in kernels/utils.jl
