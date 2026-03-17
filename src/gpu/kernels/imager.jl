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
Modified Bessel function I₀(x) — Abramowitz & Stegun polynomial approximation.
Accurate to <2e-7 for all x ≥ 0. GPU-safe (no special functions needed).
"""
@inline function besseli0_approx(x::Float64)
    ax = abs(x)
    if ax < 3.75
        t = (ax / 3.75)^2
        return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 +
               t*(0.2659732 + t*(0.0360768 + t*0.0045813)))))
    else
        t = 3.75 / ax
        return (exp(ax) / sqrt(ax)) *
               (0.39894228 + t*(0.01328592 + t*(0.00225319 +
               t*(-0.00157565 + t*(0.00916281 + t*(-0.02057706 +
               t*(0.02635537 + t*(-0.01647633 + t*0.00392377))))))))
    end
end

"""
Kaiser-Bessel gridding kernel (1D). Standard in radio interferometric imaging.
C(u) = I₀(β √(1 - (u/W)²)) / I₀(β)  for |u| ≤ W, else 0.

β = 2.34 × W gives near-optimal alias suppression.
"""
@inline function kb_value(u::Float64, W::Float64, beta::Float64, inv_i0beta::Float64)
    t = u / W
    t2 = 1.0 - t * t
    if t2 <= 0.0
        return 0.0
    end
    return besseli0_approx(beta * sqrt(t2)) * inv_i0beta
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
#                     Bilinear Interpolation Gridding Kernel                    #
#==============================================================================#

"""
CUDA kernel for bilinear interpolation gridding with Hermitian conjugate.

Each visibility scatters to 4 surrounding grid cells with weights
proportional to the sub-cell overlap area (triangle / tent function).
Only 4 atomic adds per visibility — nearly as fast as NN but O(Δx²)
interpolation error instead of O(Δx).

Correction: divide image by sinc²(η) per axis.
"""
function grid_bilinear_kernel!(
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
        
        # Bilinear: 4 surrounding cells
        iu0 = floor(Int32, u_grid)
        iu1 = iu0 + Int32(1)
        iv0 = floor(Int32, v_grid)
        iv1 = iv0 + Int32(1)
        fu = u_grid - Float64(iu0)  # fractional part, 0 to 1
        fv = v_grid - Float64(iv0)
        
        # 4 bilinear weights (partition of unity: sum = 1)
        w00 = (1.0 - fu) * (1.0 - fv)
        w10 = fu * (1.0 - fv)
        w01 = (1.0 - fu) * fv
        w11 = fu * fv
        
        # Scatter to 4 cells with bounds checks
        if iu0 >= 1 && iu0 <= N && iv0 >= 1 && iv0 <= N
            CUDA.@atomic grid_data_re[iu0, iv0, iw] += V_re * w00
            CUDA.@atomic grid_data_im[iu0, iv0, iw] += V_im * w00
            CUDA.@atomic grid_weights[iu0, iv0, iw] += w00
        end
        if iu1 >= 1 && iu1 <= N && iv0 >= 1 && iv0 <= N
            CUDA.@atomic grid_data_re[iu1, iv0, iw] += V_re * w10
            CUDA.@atomic grid_data_im[iu1, iv0, iw] += V_im * w10
            CUDA.@atomic grid_weights[iu1, iv0, iw] += w10
        end
        if iu0 >= 1 && iu0 <= N && iv1 >= 1 && iv1 <= N
            CUDA.@atomic grid_data_re[iu0, iv1, iw] += V_re * w01
            CUDA.@atomic grid_data_im[iu0, iv1, iw] += V_im * w01
            CUDA.@atomic grid_weights[iu0, iv1, iw] += w01
        end
        if iu1 >= 1 && iu1 <= N && iv1 >= 1 && iv1 <= N
            CUDA.@atomic grid_data_re[iu1, iv1, iw] += V_re * w11
            CUDA.@atomic grid_data_im[iu1, iv1, iw] += V_im * w11
            CUDA.@atomic grid_weights[iu1, iv1, iw] += w11
        end
        
        # Hermitian conjugate: V(-u,-v,-w) = conj(V(u,v,w))
        uc_grid = -u / uv_cell + center
        vc_grid = -v / uv_cell + center
        neg_w = -w
        if Nw == 1
            iw_c = Int32(1)
        else
            iw_c = clamp(floor(Int32, (neg_w - w_min) / w_range * Nw) + 1, Int32(1), Int32(Nw))
        end
        
        iu0c = floor(Int32, uc_grid)
        iu1c = iu0c + Int32(1)
        iv0c = floor(Int32, vc_grid)
        iv1c = iv0c + Int32(1)
        fuc = uc_grid - Float64(iu0c)
        fvc = vc_grid - Float64(iv0c)
        
        wc00 = (1.0 - fuc) * (1.0 - fvc)
        wc10 = fuc * (1.0 - fvc)
        wc01 = (1.0 - fuc) * fvc
        wc11 = fuc * fvc
        
        if iu0c >= 1 && iu0c <= N && iv0c >= 1 && iv0c <= N
            CUDA.@atomic grid_data_re[iu0c, iv0c, iw_c] += V_re * wc00
            CUDA.@atomic grid_data_im[iu0c, iv0c, iw_c] -= V_im * wc00
            CUDA.@atomic grid_weights[iu0c, iv0c, iw_c] += wc00
        end
        if iu1c >= 1 && iu1c <= N && iv0c >= 1 && iv0c <= N
            CUDA.@atomic grid_data_re[iu1c, iv0c, iw_c] += V_re * wc10
            CUDA.@atomic grid_data_im[iu1c, iv0c, iw_c] -= V_im * wc10
            CUDA.@atomic grid_weights[iu1c, iv0c, iw_c] += wc10
        end
        if iu0c >= 1 && iu0c <= N && iv1c >= 1 && iv1c <= N
            CUDA.@atomic grid_data_re[iu0c, iv1c, iw_c] += V_re * wc01
            CUDA.@atomic grid_data_im[iu0c, iv1c, iw_c] -= V_im * wc01
            CUDA.@atomic grid_weights[iu0c, iv1c, iw_c] += wc01
        end
        if iu1c >= 1 && iu1c <= N && iv1c >= 1 && iv1c <= N
            CUDA.@atomic grid_data_re[iu1c, iv1c, iw_c] += V_re * wc11
            CUDA.@atomic grid_data_im[iu1c, iv1c, iw_c] -= V_im * wc11
            CUDA.@atomic grid_weights[iu1c, iv1c, iw_c] += wc11
        end
    end
    
    return nothing
end

#==============================================================================#
#                    Convolutional Gridding Kernel                             #
#==============================================================================##

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
        
        # Kaiser-Bessel kernel parameters
        W = Float64(support)
        beta = 2.34 * W
        inv_i0beta = 1.0 / besseli0_approx(beta)

        # Scatter to grid cells within support
        iu_min = max(Int32(1), floor(Int32, u_grid) - support)
        iu_max = min(Int32(N), ceil(Int32, u_grid) + support)
        iv_min = max(Int32(1), floor(Int32, v_grid) - support)
        iv_max = min(Int32(N), ceil(Int32, v_grid) + support)
        
        for iv in iv_min:iv_max
            dv = iv - v_grid
            kv = kb_value(dv, W, beta, inv_i0beta)
            if kv == 0.0; continue; end
            for iu in iu_min:iu_max
                du = iu - u_grid
                ku = kb_value(du, W, beta, inv_i0beta)
                kernel_weight = ku * kv
                if kernel_weight == 0.0; continue; end
                
                CUDA.@atomic grid_data_re[iu, iv, iw] += V_re * kernel_weight
                CUDA.@atomic grid_data_im[iu, iv, iw] += V_im * kernel_weight
                CUDA.@atomic grid_weights[iu, iv, iw] += kernel_weight
            end
        end

        # Hermitian conjugate: V(-u,-v,-w) = conj(V(u,v,w))
        u_conj_grid = -u / uv_cell + center
        v_conj_grid = -v / uv_cell + center
        neg_w = -w
        if Nw == 1
            iw_conj = Int32(1)
        else
            iw_conj = clamp(floor(Int32, (neg_w - w_min) / w_range * Nw) + 1, Int32(1), Int32(Nw))
        end

        iu_min_c = max(Int32(1), floor(Int32, u_conj_grid) - support)
        iu_max_c = min(Int32(N), ceil(Int32, u_conj_grid) + support)
        iv_min_c = max(Int32(1), floor(Int32, v_conj_grid) - support)
        iv_max_c = min(Int32(N), ceil(Int32, v_conj_grid) + support)

        for iv in iv_min_c:iv_max_c
            dv = iv - v_conj_grid
            kv = kb_value(dv, W, beta, inv_i0beta)
            if kv == 0.0; continue; end
            for iu in iu_min_c:iu_max_c
                du = iu - u_conj_grid
                ku = kb_value(du, W, beta, inv_i0beta)
                kernel_weight = ku * kv
                if kernel_weight == 0.0; continue; end

                CUDA.@atomic grid_data_re[iu, iv, iw_conj] += V_re * kernel_weight
                CUDA.@atomic grid_data_im[iu, iv, iw_conj] -= V_im * kernel_weight  # Conjugate
                CUDA.@atomic grid_weights[iu, iv, iw_conj] += kernel_weight
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
