# GPU Corrupt/ApplyCal Kernels
# These kernels apply or corrupt visibilities with calibration solutions

using CUDA

#==============================================================================#
#                           Helper Functions                                   #
#==============================================================================#

"""
Check if an array is actually on GPU (CuArray) vs CPU (regular Array)
"""
is_gpu_array(x::AbstractArray) = x isa CuArray
is_gpu_array(x) = false

"""
Check if GPUVisibilities data is on GPU
"""
function data_on_gpu(vis::GPUVisibilities)
    return is_gpu_array(vis.xx)
end

#==============================================================================#
#                           CUDA Kernels                                       #
#==============================================================================#

"""
CUDA kernel to corrupt visibilities: V = J₁ * V * J₂'

Each thread handles one (baseline, frequency) pair.
"""
function corrupt_kernel!(
    # Output (modified in place)
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Calibration gains (Nant, Nfreq)
    cal_xx, cal_xy, cal_yx, cal_yy, cal_flags,
    # Baseline indices (2, Nbase)
    baselines,
    # Dimensions
    Nbase, Nfreq, is_diagonal
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        # Convert linear index to (α, β) = (baseline, frequency)
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        # Get antenna indices (1-indexed)
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Get calibration Jones matrices for both antennas
        j1_xx = cal_xx[ant1, β]
        j1_yy = cal_yy[ant1, β]
        j2_xx = cal_xx[ant2, β]
        j2_yy = cal_yy[ant2, β]
        
        if is_diagonal
            # Diagonal case: J₁ = diag(j1_xx, j1_yy)
            # V_new = J₁ * V * J₂'
            # For diagonal J: this is element-wise multiplication
            
            vxx = vis_xx[α, β]
            vxy = vis_xy[α, β]
            vyx = vis_yx[α, β]
            vyy = vis_yy[α, β]
            
            # J₁ * V
            t_xx = j1_xx * vxx
            t_xy = j1_xx * vxy
            t_yx = j1_yy * vyx
            t_yy = j1_yy * vyy
            
            # (J₁ * V) * J₂' = (J₁ * V) * diag(conj(j2_xx), conj(j2_yy))
            vis_xx[α, β] = t_xx * conj(j2_xx)
            vis_xy[α, β] = t_xy * conj(j2_yy)
            vis_yx[α, β] = t_yx * conj(j2_xx)
            vis_yy[α, β] = t_yy * conj(j2_yy)
        else
            # Full Jones matrix case
            j1_xy = cal_xy[ant1, β]
            j1_yx = cal_yx[ant1, β]
            j2_xy = cal_xy[ant2, β]
            j2_yx = cal_yx[ant2, β]
            
            vxx = vis_xx[α, β]
            vxy = vis_xy[α, β]
            vyx = vis_yx[α, β]
            vyy = vis_yy[α, β]
            
            # J₁ * V
            t_xx, t_xy, t_yx, t_yy = jones_multiply(
                j1_xx, j1_xy, j1_yx, j1_yy,
                vxx, vxy, vyx, vyy
            )
            
            # (J₁ * V) * J₂'
            r_xx, r_xy, r_yx, r_yy = jones_multiply_conjtrans(
                t_xx, t_xy, t_yx, t_yy,
                j2_xx, j2_xy, j2_yx, j2_yy
            )
            
            vis_xx[α, β] = r_xx
            vis_xy[α, β] = r_xy
            vis_yx[α, β] = r_yx
            vis_yy[α, β] = r_yy
        end
    end
    
    return nothing
end


"""
CUDA kernel to apply calibration (inverse corrupt): V = J₁⁻¹ * V * (J₂⁻¹)'
"""
function applycal_kernel!(
    # Output (modified in place)
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Calibration gains (Nant, Nfreq)
    cal_xx, cal_xy, cal_yx, cal_yy, cal_flags,
    # Baseline indices (2, Nbase)
    baselines,
    # Dimensions
    Nbase, Nfreq, is_diagonal
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        j1_xx = cal_xx[ant1, β]
        j1_yy = cal_yy[ant1, β]
        j2_xx = cal_xx[ant2, β]
        j2_yy = cal_yy[ant2, β]
        
        if is_diagonal
            # Inverse of diagonal: just reciprocal
            inv_j1_xx = 1.0 / j1_xx
            inv_j1_yy = 1.0 / j1_yy
            inv_j2_xx = 1.0 / j2_xx
            inv_j2_yy = 1.0 / j2_yy
            
            vxx = vis_xx[α, β]
            vxy = vis_xy[α, β]
            vyx = vis_yx[α, β]
            vyy = vis_yy[α, β]
            
            # J₁⁻¹ * V
            t_xx = inv_j1_xx * vxx
            t_xy = inv_j1_xx * vxy
            t_yx = inv_j1_yy * vyx
            t_yy = inv_j1_yy * vyy
            
            # (J₁⁻¹ * V) * (J₂⁻¹)'
            vis_xx[α, β] = t_xx * conj(inv_j2_xx)
            vis_xy[α, β] = t_xy * conj(inv_j2_yy)
            vis_yx[α, β] = t_yx * conj(inv_j2_xx)
            vis_yy[α, β] = t_yy * conj(inv_j2_yy)
        else
            # Full matrix inverse
            j1_xy = cal_xy[ant1, β]
            j1_yx = cal_yx[ant1, β]
            j2_xy = cal_xy[ant2, β]
            j2_yx = cal_yx[ant2, β]
            
            inv1_xx, inv1_xy, inv1_yx, inv1_yy = jones_inv(j1_xx, j1_xy, j1_yx, j1_yy)
            inv2_xx, inv2_xy, inv2_yx, inv2_yy = jones_inv(j2_xx, j2_xy, j2_yx, j2_yy)
            
            vxx = vis_xx[α, β]
            vxy = vis_xy[α, β]
            vyx = vis_yx[α, β]
            vyy = vis_yy[α, β]
            
            # J₁⁻¹ * V
            t_xx, t_xy, t_yx, t_yy = jones_multiply(
                inv1_xx, inv1_xy, inv1_yx, inv1_yy,
                vxx, vxy, vyx, vyy
            )
            
            # (J₁⁻¹ * V) * (J₂⁻¹)'
            r_xx, r_xy, r_yx, r_yy = jones_multiply_conjtrans(
                t_xx, t_xy, t_yx, t_yy,
                inv2_xx, inv2_xy, inv2_yx, inv2_yy
            )
            
            vis_xx[α, β] = r_xx
            vis_xy[α, β] = r_xy
            vis_yx[α, β] = r_yx
            vis_yy[α, β] = r_yy
        end
    end
    
    return nothing
end


#==============================================================================#
#                           High-Level Functions                               #
#==============================================================================#

"""
    gpu_corrupt!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)

GPU-accelerated corruption of visibilities.
Computes V = J₁ * V * J₂' for each baseline.
Falls back to CPU if CUDA is not available.
"""
function gpu_corrupt!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    # Use CPU fallback if CUDA is not available OR if data is on CPU
    if !CUDA.functional() || !data_on_gpu(vis)
        return cpu_corrupt!(vis, cal, meta)
    end
    
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    @cuda blocks=blocks threads=threads corrupt_kernel!(
        vis.xx, vis.xy, vis.yx, vis.yy,
        cal.xx, cal.xy, cal.yx, cal.yy, cal.flags,
        meta.baselines,
        Nb, Nf, cal.is_diagonal
    )
    
    CUDA.synchronize()
    vis
end

"""
    gpu_applycal!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)

GPU-accelerated application of calibration.
Computes V = J₁⁻¹ * V * (J₂⁻¹)' for each baseline.
Falls back to CPU if CUDA is not available.
"""
function gpu_applycal!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    # Use CPU fallback if CUDA is not available OR if data is on CPU
    if !CUDA.functional() || !data_on_gpu(vis)
        return cpu_applycal!(vis, cal, meta)
    end
    
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    @cuda blocks=blocks threads=threads applycal_kernel!(
        vis.xx, vis.xy, vis.yx, vis.yy,
        cal.xx, cal.xy, cal.yx, cal.yy, cal.flags,
        meta.baselines,
        Nb, Nf, cal.is_diagonal
    )
    
    CUDA.synchronize()
    vis
end


#==============================================================================#
#                           CPU Fallback Functions                             #
#==============================================================================#

"""
CPU fallback for corrupt!
"""
function cpu_corrupt!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    baselines = meta.baselines
    
    @inbounds for β in 1:Nf
        for α in 1:Nb
            ant1 = baselines[1, α]
            ant2 = baselines[2, α]
            
            j1_xx = cal.xx[ant1, β]
            j1_yy = cal.yy[ant1, β]
            j2_xx = cal.xx[ant2, β]
            j2_yy = cal.yy[ant2, β]
            
            if cal.is_diagonal
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                vis.xx[α, β] = j1_xx * vxx * conj(j2_xx)
                vis.xy[α, β] = j1_xx * vxy * conj(j2_yy)
                vis.yx[α, β] = j1_yy * vyx * conj(j2_xx)
                vis.yy[α, β] = j1_yy * vyy * conj(j2_yy)
            else
                j1_xy = cal.xy[ant1, β]
                j1_yx = cal.yx[ant1, β]
                j2_xy = cal.xy[ant2, β]
                j2_yx = cal.yx[ant2, β]
                
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                # J₁ * V
                t_xx = j1_xx*vxx + j1_xy*vyx
                t_xy = j1_xx*vxy + j1_xy*vyy
                t_yx = j1_yx*vxx + j1_yy*vyx
                t_yy = j1_yx*vxy + j1_yy*vyy
                
                # (J₁ * V) * J₂'
                vis.xx[α, β] = t_xx*conj(j2_xx) + t_xy*conj(j2_xy)
                vis.xy[α, β] = t_xx*conj(j2_yx) + t_xy*conj(j2_yy)
                vis.yx[α, β] = t_yx*conj(j2_xx) + t_yy*conj(j2_xy)
                vis.yy[α, β] = t_yx*conj(j2_yx) + t_yy*conj(j2_yy)
            end
        end
    end
    vis
end
"""
CPU fallback for applycal!
"""
function cpu_applycal!(vis::GPUVisibilities, cal::GPUCalibration, meta::GPUMetadata)
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    baselines = meta.baselines
    
    @inbounds for β in 1:Nf
        for α in 1:Nb
            ant1 = baselines[1, α]
            ant2 = baselines[2, α]
            
            j1_xx = cal.xx[ant1, β]
            j1_yy = cal.yy[ant1, β]
            j2_xx = cal.xx[ant2, β]
            j2_yy = cal.yy[ant2, β]
            
            # Skip flagged antennas
            if cal.flags[ant1, β] || cal.flags[ant2, β]
                vis.flags[α, β] = true
                continue
            end
            
            if cal.is_diagonal
                # Invert diagonal Jones matrices
                inv_j1_xx = 1.0 / j1_xx
                inv_j1_yy = 1.0 / j1_yy
                inv_j2_xx = 1.0 / j2_xx
                inv_j2_yy = 1.0 / j2_yy
                
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                vis.xx[α, β] = inv_j1_xx * vxx * conj(inv_j2_xx)
                vis.xy[α, β] = inv_j1_xx * vxy * conj(inv_j2_yy)
                vis.yx[α, β] = inv_j1_yy * vyx * conj(inv_j2_xx)
                vis.yy[α, β] = inv_j1_yy * vyy * conj(inv_j2_yy)
            else
                j1_xy = cal.xy[ant1, β]
                j1_yx = cal.yx[ant1, β]
                j2_xy = cal.xy[ant2, β]
                j2_yx = cal.yx[ant2, β]
                
                # Compute inverse of J1 (2x2 matrix inverse)
                det1 = j1_xx*j1_yy - j1_xy*j1_yx
                inv_j1_xx = j1_yy / det1
                inv_j1_xy = -j1_xy / det1
                inv_j1_yx = -j1_yx / det1
                inv_j1_yy = j1_xx / det1
                
                # Compute inverse of J2
                det2 = j2_xx*j2_yy - j2_xy*j2_yx
                inv_j2_xx = j2_yy / det2
                inv_j2_xy = -j2_xy / det2
                inv_j2_yx = -j2_yx / det2
                inv_j2_yy = j2_xx / det2
                
                vxx = vis.xx[α, β]
                vxy = vis.xy[α, β]
                vyx = vis.yx[α, β]
                vyy = vis.yy[α, β]
                
                # J₁⁻¹ * V
                t_xx = inv_j1_xx*vxx + inv_j1_xy*vyx
                t_xy = inv_j1_xx*vxy + inv_j1_xy*vyy
                t_yx = inv_j1_yx*vxx + inv_j1_yy*vyx
                t_yy = inv_j1_yx*vxy + inv_j1_yy*vyy
                
                # (J₁⁻¹ * V) * (J₂⁻¹)'
                vis.xx[α, β] = t_xx*conj(inv_j2_xx) + t_xy*conj(inv_j2_xy)
                vis.xy[α, β] = t_xx*conj(inv_j2_yx) + t_xy*conj(inv_j2_yy)
                vis.yx[α, β] = t_yx*conj(inv_j2_xx) + t_yy*conj(inv_j2_xy)
                vis.yy[α, β] = t_yx*conj(inv_j2_yx) + t_yy*conj(inv_j2_yy)
            end
        end
    end
    vis
end