# GPU Imaging Kernels
#
# CUDA kernels for visibility gridding and w-term correction.
# Optimized for wide-field imaging with w-stacking.
#
# These kernels are included directly into the TTCalX module (not a submodule),
# so they access thread_index_1d(), etc. from the parent scope.

const GPU_C_IMAGER = 2.99792458e8  # Speed of light m/s

#==============================================================================#
#                     Gridding Convolution Functions                           #
#==============================================================================#

"""
Prolate spheroidal wave function (PSWF) approximation for gridding convolution.
Uses a Gaussian approximation.

# Arguments
- `u`: normalized coordinate (-1 to 1)
- `support`: kernel half-width
"""
@inline function pswf_value(u::Float64, support::Int=3)
    sigma = support / 2.5
    return exp(-u^2 / (2 * sigma^2))
end

"""
Compute gridding convolution kernel weight for a single point.
"""
@inline function gridding_kernel_weight(du::Float64, dv::Float64, support::Int)
    if abs(du) > support || abs(dv) > support
        return 0.0
    end
    return pswf_value(du / support) * pswf_value(dv / support)
end

#==============================================================================#
#                        Nearest-Neighbor Gridding Kernel                       #
#==============================================================================#

"""
CUDA kernel for nearest-neighbor gridding with Hermitian conjugate.

Each thread handles one (baseline, frequency) pair.
Uses atomic adds for thread-safe accumulation onto the shared UV grid.
"""
function grid_nn_kernel!(
    # Output grid (N, N, Nw)
    grid_data_re, grid_data_im,
    grid_weights,
    # Input visibilities (Nbase, Nfreq)
    vis_xx_re, vis_xx_im,
    vis_yy_re, vis_yy_im,
    flags,
    # UVW coordinates (3, Nbase)
    uvw,
    # Frequency channels (Nfreq,)
    channels,
    # W-layer info
    w_min, w_range, Nw,
    # Grid parameters
    N, uv_cell, center,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        # Check flag
        if flags[α, β]
            return nothing
        end
        
        # Wavelength
        ν = channels[β]
        λ = GPU_C_IMAGER / ν
        
        # UVW in wavelengths
        u = uvw[1, α] / λ
        v = uvw[2, α] / λ
        w = uvw[3, α] / λ
        
        # Grid indices (nearest neighbor)
        iu = round(Int32, u / uv_cell) + center
        iv = round(Int32, v / uv_cell) + center
        
        # Bounds check
        if iu < 1 || iu > N || iv < 1 || iv > N
            return nothing
        end
        
        # W-layer index
        if Nw == 1
            iw = Int32(1)
        else
            iw = clamp(floor(Int32, (w - w_min) / w_range * Nw) + 1, Int32(1), Int32(Nw))
        end
        
        # Stokes I visibility: (XX + YY) / 2
        V_re = 0.5 * (vis_xx_re[α, β] + vis_yy_re[α, β])
        V_im = 0.5 * (vis_xx_im[α, β] + vis_yy_im[α, β])
        
        # Atomic add to grid
        CUDA.@atomic grid_data_re[iu, iv, iw] += V_re
        CUDA.@atomic grid_data_im[iu, iv, iw] += V_im
        CUDA.@atomic grid_weights[iu, iv, iw] += 1.0
        
        # Hermitian conjugate: V(-u,-v,-w) = conj(V(u,v,w))
        iu_conj = N - iu + 2
        iv_conj = N - iv + 2
        if iu_conj >= 1 && iu_conj <= N && iv_conj >= 1 && iv_conj <= N
            neg_w = -w
            if Nw == 1
                iw_conj = Int32(1)
            else
                iw_conj = clamp(floor(Int32, (neg_w - w_min) / w_range * Nw) + 1, Int32(1), Int32(Nw))
            end
            CUDA.@atomic grid_data_re[iu_conj, iv_conj, iw_conj] += V_re
            CUDA.@atomic grid_data_im[iu_conj, iv_conj, iw_conj] -= V_im  # Conjugate
            CUDA.@atomic grid_weights[iu_conj, iv_conj, iw_conj] += 1.0
        end
    end
    
    return nothing
end

#==============================================================================#
#                    Convolutional Gridding Kernel                             #
#==============================================================================#

"""
CUDA kernel for gridding with convolution kernel (more accurate, slower).

Each thread handles one (baseline, frequency) pair and scatters
to nearby grid cells within the support radius.
"""
function grid_convolve_kernel!(
    # Output grid (N, N, Nw)
    grid_data_re, grid_data_im,
    grid_weights,
    # Input visibilities (Nbase, Nfreq)
    vis_xx_re, vis_xx_im,
    vis_yy_re, vis_yy_im,
    flags,
    # UVW coordinates (3, Nbase)
    uvw,
    # Frequency channels (Nfreq,)
    channels,
    # W-layer info
    w_min, w_range, Nw,
    # Grid parameters
    N, uv_cell, center,
    # Convolution support (half-width)
    support,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        if flags[α, β]
            return nothing
        end
        
        ν = channels[β]
        λ = GPU_C_IMAGER / ν
        
        u = uvw[1, α] / λ
        v = uvw[2, α] / λ
        w = uvw[3, α] / λ
        
        # Continuous grid position
        u_grid = u / uv_cell + center
        v_grid = v / uv_cell + center
        
        # W-layer index
        if Nw == 1
            iw = Int32(1)
        else
            iw = clamp(floor(Int32, (w - w_min) / w_range * Nw) + 1, Int32(1), Int32(Nw))
        end
        
        # Stokes I visibility
        V_re = 0.5 * (vis_xx_re[α, β] + vis_yy_re[α, β])
        V_im = 0.5 * (vis_xx_im[α, β] + vis_yy_im[α, β])
        
        # Scatter to grid cells within support
        iu_min = max(Int32(1), floor(Int32, u_grid) - support)
        iu_max = min(Int32(N), ceil(Int32, u_grid) + support)
        iv_min = max(Int32(1), floor(Int32, v_grid) - support)
        iv_max = min(Int32(N), ceil(Int32, v_grid) + support)
        
        for iv in iv_min:iv_max
            dv = iv - v_grid
            for iu in iu_min:iu_max
                du = iu - u_grid
                
                r2 = du^2 + dv^2
                if r2 > support^2
                    continue
                end
                
                # Gaussian kernel weight
                kernel_weight = exp(-r2 / (2.0 * (support/2.5)^2))
                
                CUDA.@atomic grid_data_re[iu, iv, iw] += V_re * kernel_weight
                CUDA.@atomic grid_data_im[iu, iv, iw] += V_im * kernel_weight
                CUDA.@atomic grid_weights[iu, iv, iw] += kernel_weight
            end
        end
    end
    
    return nothing
end

#==============================================================================#
#                      W-Correction Kernel                                      #
#==============================================================================#

"""
CUDA kernel to apply w-correction phase to image pixels.

The w-term phase shift is: exp(2πi w (√(1-l²-m²) - 1))

Each thread handles one pixel. Results are atomically accumulated
into the output image across w-layers.
"""
function w_correction_kernel!(
    # Output image (N, N) - accumulated
    image_re, image_im,
    # Input FFT'd layer (N, N)
    layer_re, layer_im,
    # W-value for this layer (wavelengths)
    w_layer,
    # Image parameters
    N, cell,
    # Pre-computed center index
    center
)
    idx = thread_index_1d()
    
    if idx <= N * N
        iu = ((idx - 1) % N) + 1
        iv = ((idx - 1) ÷ N) + 1
        
        # l, m coordinates
        l = (iu - center) * cell
        m = (iv - center) * cell
        
        r2 = l^2 + m^2
        if r2 >= 1.0
            return nothing
        end
        
        # n = sqrt(1 - l² - m²)
        n = sqrt(1.0 - r2)
        
        # W-correction phase: exp(2πi w (n-1))
        phase = 2π * w_layer * (n - 1.0)
        cos_phi = cos(phase)
        sin_phi = sin(phase)
        
        # Complex multiply: (layer_re + i*layer_im) * (cos_phi + i*sin_phi)
        val_re = layer_re[iu, iv]
        val_im = layer_im[iu, iv]
        
        corrected_re = val_re * cos_phi - val_im * sin_phi
        corrected_im = val_re * sin_phi + val_im * cos_phi
        
        # Atomic add to output
        CUDA.@atomic image_re[iu, iv] += corrected_re
        CUDA.@atomic image_im[iu, iv] += corrected_im
    end
    
    return nothing
end

#==============================================================================#
#                       Normalization Kernel                                    #
#==============================================================================#

"""
CUDA kernel to normalize gridded visibilities by weights.
"""
function normalize_grid_kernel!(
    grid_re, grid_im,
    weights,
    N, Nw
)
    idx = thread_index_1d()
    
    if idx <= N * N * Nw
        iu = ((idx - 1) % N) + 1
        rest = (idx - 1) ÷ N
        iv = (rest % N) + 1
        iw = (rest ÷ N) + 1
        
        w = weights[iu, iv, iw]
        if w > 0
            grid_re[iu, iv, iw] /= w
            grid_im[iu, iv, iw] /= w
        end
    end
    
    return nothing
end
