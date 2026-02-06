# GPU Stefcal Kernels
# Main calibration algorithm - GPU-accelerated version of stefcal_step

using CUDA

#==============================================================================#
#                        Makesquare Kernel                                     #
#==============================================================================#

"""
CUDA kernel to reorganize visibility data into square matrices.
Converts from baseline-indexed to antenna-antenna indexed format.
"""
function makesquare_kernel!(
    # Output square matrices (Nant, Nant, Nfreq)
    sq_xx, sq_xy, sq_yx, sq_yy,
    # Input visibility data (Nbase, Nfreq)
    vis_xx, vis_xy, vis_yx, vis_yy, vis_flags,
    # Baseline indices (2, Nbase)
    baselines,
    # Dimensions
    Nant, Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        # Skip if flagged
        if vis_flags[α, β]
            return nothing
        end
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Skip autocorrelations
        if ant1 == ant2
            return nothing
        end
        
        vxx = vis_xx[α, β]
        vxy = vis_xy[α, β]
        vyx = vis_yx[α, β]
        vyy = vis_yy[α, β]
        
        # Fill both triangles (Hermitian)
        sq_xx[ant1, ant2, β] = vxx
        sq_xy[ant1, ant2, β] = vxy
        sq_yx[ant1, ant2, β] = vyx
        sq_yy[ant1, ant2, β] = vyy
        
        # Conjugate transpose for other triangle
        sq_xx[ant2, ant1, β] = conj(vxx)
        sq_xy[ant2, ant1, β] = conj(vyx)  # Note: xy and yx swap
        sq_yx[ant2, ant1, β] = conj(vxy)
        sq_yy[ant2, ant1, β] = conj(vyy)
    end
    
    return nothing
end


#==============================================================================#
#                      Stefcal Step Kernel (Diagonal)                          #
#==============================================================================#

"""
CUDA kernel for one step of diagonal stefcal.
Each thread computes the update for one (antenna, frequency) pair.

For diagonal Jones matrices, the update equation is:
    step[j] = (∑ᵢ (G[i]*M[i,j])' * V[i,j]) / (∑ᵢ (G[i]*M[i,j])' * (G[i]*M[i,j]))

Where the operations are for diagonal matrices.
"""
function stefcal_step_diagonal_kernel!(
    # Output step (Nant, Nfreq)
    step_xx, step_yy,
    # Current gains (Nant, Nfreq)
    gain_xx, gain_yy,
    # Square measured visibilities (Nant, Nant, Nfreq)
    meas_xx, meas_xy, meas_yx, meas_yy,
    # Square model visibilities (Nant, Nant, Nfreq)
    model_xx, model_xy, model_yx, model_yy,
    # Dimensions
    Nant, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1   # antenna index
        β = ((idx - 1) ÷ Nant) + 1   # frequency index
        
        # Accumulators for numerator and denominator (diagonal Jones)
        num_xx = ComplexF64(0)
        num_yy = ComplexF64(0)
        den_xx = ComplexF64(0)
        den_yy = ComplexF64(0)
        
        # Sum over all antennas i
        for i in 1:Nant
            # Get current gain for antenna i
            gi_xx = gain_xx[i, β]
            gi_yy = gain_yy[i, β]
            
            # Get model visibility M[i,j]
            m_xx = model_xx[i, j, β]
            m_xy = model_xy[i, j, β]
            m_yx = model_yx[i, j, β]
            m_yy = model_yy[i, j, β]
            
            # Get measured visibility V[i,j]
            v_xx = meas_xx[i, j, β]
            v_xy = meas_xy[i, j, β]
            v_yx = meas_yx[i, j, β]
            v_yy = meas_yy[i, j, β]
            
            # GM = G[i] * M[i,j] where G is diagonal
            gm_xx = gi_xx * m_xx
            gm_xy = gi_xx * m_xy
            gm_yx = gi_yy * m_yx
            gm_yy = gi_yy * m_yy
            
            # inner_multiply for DiagonalJonesMatrix:
            # numerator += (GM.xx'*V.xx + GM.yx'*V.yx, GM.xy'*V.xy + GM.yy'*V.yy)
            num_xx += conj(gm_xx) * v_xx + conj(gm_yx) * v_yx
            num_yy += conj(gm_xy) * v_xy + conj(gm_yy) * v_yy
            
            # denominator += (GM.xx'*GM.xx + GM.yx'*GM.yx, GM.xy'*GM.xy + GM.yy'*GM.yy)
            den_xx += conj(gm_xx) * gm_xx + conj(gm_yx) * gm_yx
            den_yy += conj(gm_xy) * gm_xy + conj(gm_yy) * gm_yy
        end
        
        # Compute step: step = (denominator \ numerator)' - current
        # For diagonal: this is element-wise division
        ok_xx = abs(den_xx) > eps(Float64)
        ok_yy = abs(den_yy) > eps(Float64)
        
        new_xx = ok_xx ? conj(num_xx / den_xx) : gain_xx[j, β]
        new_yy = ok_yy ? conj(num_yy / den_yy) : gain_yy[j, β]
        
        step_xx[j, β] = new_xx - gain_xx[j, β]
        step_yy[j, β] = new_yy - gain_yy[j, β]
    end
    
    return nothing
end


#==============================================================================#
#                      Stefcal Step Kernel (Full)                              #
#==============================================================================#

"""
CUDA kernel for one step of full Jones stefcal (used in zesting).
"""
function stefcal_step_full_kernel!(
    # Output step (Nant, Nfreq)
    step_xx, step_xy, step_yx, step_yy,
    # Current gains (Nant, Nfreq)
    gain_xx, gain_xy, gain_yx, gain_yy,
    # Square measured visibilities (Nant, Nant, Nfreq)
    meas_xx, meas_xy, meas_yx, meas_yy,
    # Square model visibilities (Nant, Nant, Nfreq)
    model_xx, model_xy, model_yx, model_yy,
    # Dimensions
    Nant, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        
        # Accumulators for 2x2 numerator and denominator matrices
        num_xx = ComplexF64(0)
        num_xy = ComplexF64(0)
        num_yx = ComplexF64(0)
        num_yy = ComplexF64(0)
        
        den_xx = ComplexF64(0)
        den_xy = ComplexF64(0)
        den_yx = ComplexF64(0)
        den_yy = ComplexF64(0)
        
        for i in 1:Nant
            # Get current gain for antenna i (full Jones)
            gi_xx = gain_xx[i, β]
            gi_xy = gain_xy[i, β]
            gi_yx = gain_yx[i, β]
            gi_yy = gain_yy[i, β]
            
            # Get model visibility M[i,j]
            m_xx = model_xx[i, j, β]
            m_xy = model_xy[i, j, β]
            m_yx = model_yx[i, j, β]
            m_yy = model_yy[i, j, β]
            
            # Get measured visibility V[i,j]
            v_xx = meas_xx[i, j, β]
            v_xy = meas_xy[i, j, β]
            v_yx = meas_yx[i, j, β]
            v_yy = meas_yy[i, j, β]
            
            # GM = G[i] * M[i,j] (full matrix multiply)
            gm_xx = gi_xx*m_xx + gi_xy*m_yx
            gm_xy = gi_xx*m_xy + gi_xy*m_yy
            gm_yx = gi_yx*m_xx + gi_yy*m_yx
            gm_yy = gi_yx*m_xy + gi_yy*m_yy
            
            # inner_multiply for JonesMatrix: GM' * V
            # This is the conjugate transpose of GM times V
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
        
        # Solve: step = (denominator \ numerator)' - current
        # This requires 2x2 matrix solve
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
            new_xy = conj(sol_yx)  # Note: xy and yx swap in transpose
            new_yx = conj(sol_xy)
            new_yy = conj(sol_yy)
            
            step_xx[j, β] = new_xx - gain_xx[j, β]
            step_xy[j, β] = new_xy - gain_xy[j, β]
            step_yx[j, β] = new_yx - gain_yx[j, β]
            step_yy[j, β] = new_yy - gain_yy[j, β]
        else
            step_xx[j, β] = ComplexF64(0)
            step_xy[j, β] = ComplexF64(0)
            step_yx[j, β] = ComplexF64(0)
            step_yy[j, β] = ComplexF64(0)
        end
    end
    
    return nothing
end


#==============================================================================#
#                        Update Gains Kernel                                   #
#==============================================================================#

"""
Kernel to update gains: gain += step
Also computes step norm for convergence check.
"""
function update_gains_kernel!(
    # Gains to update (Nant, Nfreq)
    gain_xx, gain_xy, gain_yx, gain_yy,
    # Step values (Nant, Nfreq)
    step_xx, step_xy, step_yx, step_yy,
    # Norm accumulators (Nfreq,) - one per frequency channel
    step_norms, gain_norms,
    # Dimensions
    Nant, Nfreq, is_diagonal
)
    idx = thread_index_1d()
    
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        
        # Update gains
        gain_xx[j, β] += step_xx[j, β]
        gain_yy[j, β] += step_yy[j, β]
        
        if !is_diagonal
            gain_xy[j, β] += step_xy[j, β]
            gain_yx[j, β] += step_yx[j, β]
        end
        
        # Accumulate norms (atomic add for thread safety)
        if is_diagonal
            s_norm = abs2(step_xx[j, β]) + abs2(step_yy[j, β])
            g_norm = abs2(gain_xx[j, β]) + abs2(gain_yy[j, β])
        else
            s_norm = abs2(step_xx[j, β]) + abs2(step_xy[j, β]) + 
                     abs2(step_yx[j, β]) + abs2(step_yy[j, β])
            g_norm = abs2(gain_xx[j, β]) + abs2(gain_xy[j, β]) + 
                     abs2(gain_yx[j, β]) + abs2(gain_yy[j, β])
        end
        
        CUDA.@atomic step_norms[β] += s_norm
        CUDA.@atomic gain_norms[β] += g_norm
    end
    
    return nothing
end


#==============================================================================#
#                        High-Level Functions                                  #
#==============================================================================#

"""
    gpu_makesquare!(sq_meas, sq_model, vis_meas, vis_model, meta)

Convert baseline-indexed visibilities to antenna-antenna square matrices.
"""
function gpu_makesquare!(sq_meas::GPUSquareVisibilities, sq_model::GPUSquareVisibilities,
                         vis_meas::GPUVisibilities, vis_model::GPUVisibilities,
                         meta::GPUMetadata)
    Na = Nant(meta)
    Nb = Nbase(meta)
    Nf = Nfreq(meta)
    
    # Zero output arrays
    sq_meas.xx .= 0
    sq_meas.xy .= 0
    sq_meas.yx .= 0
    sq_meas.yy .= 0
    sq_model.xx .= 0
    sq_model.xy .= 0
    sq_model.yx .= 0
    sq_model.yy .= 0
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    # Process measured visibilities
    @cuda blocks=blocks threads=threads makesquare_kernel!(
        sq_meas.xx, sq_meas.xy, sq_meas.yx, sq_meas.yy,
        vis_meas.xx, vis_meas.xy, vis_meas.yx, vis_meas.yy, vis_meas.flags,
        meta.baselines, Na, Nb, Nf
    )
    
    # Process model visibilities
    @cuda blocks=blocks threads=threads makesquare_kernel!(
        sq_model.xx, sq_model.xy, sq_model.yx, sq_model.yy,
        vis_model.xx, vis_model.xy, vis_model.yx, vis_model.yy, vis_model.flags,
        meta.baselines, Na, Nb, Nf
    )
    
    CUDA.synchronize()
    nothing
end


"""
    gpu_stefcal_step!(step, gain, sq_meas, sq_model; diagonal=true)

Compute one stefcal iteration step on GPU.
"""
function gpu_stefcal_step!(step::GPUCalibration, gain::GPUCalibration,
                           sq_meas::GPUSquareVisibilities, sq_model::GPUSquareVisibilities)
    Na = Nant(gain)
    Nf = Nfreq(gain)
    
    blocks, threads = kernel_config_1d(Na * Nf)
    
    if gain.is_diagonal
        @cuda blocks=blocks threads=threads stefcal_step_diagonal_kernel!(
            step.xx, step.yy,
            gain.xx, gain.yy,
            sq_meas.xx, sq_meas.xy, sq_meas.yx, sq_meas.yy,
            sq_model.xx, sq_model.xy, sq_model.yx, sq_model.yy,
            Na, Nf
        )
    else
        @cuda blocks=blocks threads=threads stefcal_step_full_kernel!(
            step.xx, step.xy, step.yx, step.yy,
            gain.xx, gain.xy, gain.yx, gain.yy,
            sq_meas.xx, sq_meas.xy, sq_meas.yx, sq_meas.yy,
            sq_model.xx, sq_model.xy, sq_model.yx, sq_model.yy,
            Na, Nf
        )
    end
    
    CUDA.synchronize()
    step
end


"""
    gpu_iterate!(gain, sq_meas, sq_model, pool; maxiter=20, tolerance=1e-3)

Iteratively solve for gains using GPU-accelerated stefcal.
Returns true if converged.
"""
function gpu_iterate!(gain::GPUCalibration, sq_meas::GPUSquareVisibilities, 
                      sq_model::GPUSquareVisibilities, pool::GPUMemoryPool;
                      maxiter::Int=20, tolerance::Float64=1e-3)
    Na = Nant(gain)
    Nf = Nfreq(gain)
    
    step = pool.step_buffer
    
    # Allocate norm tracking arrays
    step_norms = CUDA.zeros(Float64, Nf)
    gain_norms = CUDA.zeros(Float64, Nf)
    
    blocks, threads = kernel_config_1d(Na * Nf)
    
    converged = false
    for iter in 1:maxiter
        # Compute step
        gpu_stefcal_step!(step, gain, sq_meas, sq_model)
        
        # Reset norm accumulators
        step_norms .= 0
        gain_norms .= 0
        
        # Update gains and compute norms
        @cuda blocks=blocks threads=threads update_gains_kernel!(
            gain.xx, gain.xy, gain.yx, gain.yy,
            step.xx, step.xy, step.yx, step.yy,
            step_norms, gain_norms,
            Na, Nf, gain.is_diagonal
        )
        
        CUDA.synchronize()
        
        # Check convergence (on CPU)
        total_step_norm = sqrt(sum(Array(step_norms)))
        total_gain_norm = sqrt(sum(Array(gain_norms)))
        
        if total_step_norm < tolerance * total_gain_norm
            converged = true
            break
        end
    end
    
    converged
end
