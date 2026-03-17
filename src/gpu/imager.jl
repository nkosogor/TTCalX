# GPU Wide-Field Imager Module
# 
# Implements GPU-accelerated imaging with w-term correction using w-stacking.
# Designed to work directly with peeled visibilities on GPU without requiring
# intermediate MS files.
#
# Phase convention (matching CASA / wsclean):
#   V(u,v,w) = ∫∫ I(l,m) · exp(-2πi(ul + vm + w(n-1))) dl dm / n
#   I(l,m) ≈ Σ_k V_k · exp(+2πi(u_k·l + v_k·m + w_k·(n-1)))
#
# The w-stacking algorithm (Offringa et al. 2014):
#   1. Bin visibilities by w-value into discrete w-layers
#   2. Grid visibilities onto each w-layer's UV grid (no phase correction)
#   3. IFFT each w-layer to image domain (IFFT has the +2πi sign needed)
#   4. Apply w-correction phase: exp(+2πi · w_layer · (n-1)) per pixel
#   5. Sum corrected layers → dirty image
#
# Key choices:
#   - Uses IFFT (not FFT) so that the Fourier sign matches the imaging equation
#   - Natural weighting: no per-cell normalization (best sensitivity)
#   - Image normalization: divide by total weight sum after FFT
#   - Hermitian conjugate: V(-u,-v,-w) = conj(V(u,v,w)) for real sky
#
# Memory optimization:
#   - Grid stored as split Float64 real/imag arrays — CUDA kernels write
#     directly via atomic Float64 adds, no temporary 3D allocation needed
#   - FFT working buffers pre-allocated and reused across w-layers
#   - In-place cuFFT via plan_ifft! — zero allocation per iteration
#
# References:
#   - Cornwell et al. (2008) "W Projection"
#   - Offringa et al. (2014) "WSCLEAN: An implementation of a fast, generic 
#     wide-field imager for radio astronomy"
#   - Thompson, Moran & Swenson, "Interferometry and Synthesis in Radio Astronomy"

module GPUImager

using CUDA
using LinearAlgebra
using FFTW
using Statistics

# Import from parent module (TTCalX)
using ..TTCalX: GPUVisibilities, GPUMetadata, GPUCalibration
using ..TTCalX: Nbase, Nfreq, Nant
using ..TTCalX: GPUPeelingSource, AbstractGPUPeelingSource, peel_gpu!
using ..TTCalX: thread_index_1d
using ..TTCalX: grid_nn_kernel!, grid_convolve_kernel!, w_correction_kernel!
using ..TTCalX: log_step, log_substep, log_detail, log_success, log_warning

# Helper functions
_is_gpu(x::CuArray) = true
_is_gpu(x::AbstractArray) = false
_is_gpu(vis::GPUVisibilities) = _is_gpu(vis.xx)

"""Modified Bessel I₀ — Abramowitz & Stegun polynomial approximation (CPU)."""
function _besseli0(x::Float64)
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

"""1D Kaiser-Bessel kernel value (CPU)."""
function _kb_value(u::Float64, W::Float64, beta::Float64, inv_i0beta::Float64)
    t = u / W
    t2 = 1.0 - t * t
    t2 <= 0.0 && return 0.0
    return _besseli0(beta * sqrt(t2)) * inv_i0beta
end

_zeros(::Type{T}, dims...; gpu::Bool=true) where T = 
    gpu && CUDA.functional() ? CUDA.zeros(T, dims...) : zeros(T, dims...)

"""
In-place 2D FFT quadrant swap (fftshift / ifftshift for even-sized arrays).
Uses view-based copies that dispatch to GPU kernels on CuArrays.
"""
function fftshift_2d!(dst::AbstractMatrix, src::AbstractMatrix)
    N = size(src, 1)
    h = N ÷ 2
    dst[1:h, 1:h]     .= @view src[h+1:N, h+1:N]
    dst[h+1:N, h+1:N] .= @view src[1:h, 1:h]
    dst[1:h, h+1:N]   .= @view src[h+1:N, 1:h]
    dst[h+1:N, 1:h]   .= @view src[1:h, h+1:N]
    return dst
end

# Export public API
export GPUImagerConfig, GPUGrid, GPUImage
export make_image, grid_visibilities!, grid_to_image!
export field_of_view, image_coordinates, auto_configure_imager
export w_layer_indices, w_correction_phase, combine_w_layers
export peel_and_image

#==============================================================================#
#                          Configuration Types                                  #
#==============================================================================#

"""
    GPUImagerConfig(; kwargs...)

Configuration for GPU wide-field imager.

# Fields
- `image_size::Int`: Image dimensions (square, must be even for FFT centering)
- `cell_size::Float64`: Angular size of each pixel in radians
- `w_layers::Int`: Number of w-stacking layers
- `padding_factor::Float64`: FFT zero-padding factor (≥1.0)
- `weighting::Symbol`: Visibility weighting scheme (:natural, :uniform, :briggs)
- `robust::Float64`: Robust parameter for Briggs weighting (-2 to 2)
- `oversampling::Int`: Gridding convolution oversampling factor
- `support::Int`: Gridding convolution kernel support (half-width in grid cells)
- `w_max::Float64`: Maximum |w| in wavelengths (auto-computed if 0)
"""
struct GPUImagerConfig
    image_size::Int
    cell_size::Float64
    w_layers::Int
    padding_factor::Float64
    weighting::Symbol
    robust::Float64
    oversampling::Int
    support::Int
    w_max::Float64
    
    function GPUImagerConfig(;
        image_size::Int=512,
        cell_size::Float64=deg2rad(1.0/60.0),  # 1 arcmin default
        w_layers::Int=1,
        padding_factor::Float64=1.2,
        weighting::Symbol=:natural,
        robust::Float64=0.0,
        oversampling::Int=8,
        support::Int=0,
        w_max::Float64=0.0
    )
        @assert image_size > 0 && iseven(image_size) "image_size must be positive and even"
        @assert cell_size > 0 "cell_size must be positive"
        @assert w_layers >= 1 "w_layers must be at least 1"
        @assert padding_factor >= 1.0 "padding_factor must be >= 1.0"
        @assert weighting in [:natural, :uniform, :briggs] "weighting must be :natural, :uniform, or :briggs"
        @assert -2.0 <= robust <= 2.0 "robust must be between -2 and 2"
        
        new(image_size, cell_size, w_layers, padding_factor, weighting, 
            robust, oversampling, support, w_max)
    end
end

"""Compute padded grid size (must be even for FFT centering)."""
function _padded_size(config::GPUImagerConfig)
    p = round(Int, config.image_size * config.padding_factor)
    return iseven(p) ? p : p + 1
end

"""
    field_of_view(config::GPUImagerConfig) -> Float64

Compute the angular field of view of the image in radians.
"""
function field_of_view(config::GPUImagerConfig)
    return config.image_size * config.cell_size
end

"""
    auto_configure_imager(meta, vis; image_size=512, cell_size=nothing) -> GPUImagerConfig

Auto-configure imager based on observation parameters (UVW range, frequencies).
"""
function auto_configure_imager(meta::GPUMetadata, vis::GPUVisibilities;
                               image_size::Int=512,
                               cell_size::Union{Float64,Nothing}=nothing)
    # Get UVW on CPU
    uvw = meta.uvw isa CUDA.CuArray ? Array(meta.uvw) : meta.uvw
    channels = meta.channels isa CUDA.CuArray ? Array(meta.channels) : meta.channels
    
    c = 299792458.0
    λ_min = c / maximum(channels)
    
    # Compute maximum baseline in wavelengths
    u_max = maximum(abs.(uvw[1, :])) / λ_min
    v_max = maximum(abs.(uvw[2, :])) / λ_min
    uv_max = max(u_max, v_max)
    
    # Auto cell size: ~3 pixels per beam (wsclean default factor)
    if cell_size === nothing
        cell_size = 1.0 / (3.0 * uv_max)  # radians
    end
    
    # Compute w-range across all frequencies
    w_max_wavelengths = 0.0
    for β in 1:length(channels)
        λ = c / channels[β]
        w_max_β = maximum(abs.(uvw[3, :])) / λ
        w_max_wavelengths = max(w_max_wavelengths, w_max_β)
    end
    
    # Number of w-layers: phase error < 1 radian requires w_layers ≈ w_max * FoV^2 / 2
    fov = image_size * cell_size
    w_layers_needed = max(1, ceil(Int, w_max_wavelengths * fov^2 / 2.0))
    w_layers = min(w_layers_needed, 128)
    
    return GPUImagerConfig(
        image_size=image_size,
        cell_size=cell_size,
        w_layers=w_layers,
        w_max=w_max_wavelengths
    )
end

#==============================================================================#
#                            Grid Types                                         #
#==============================================================================#

"""
    GPUGrid(size, w_layers; gpu=true)

GPU-friendly UV grid for w-stacking using split real/imaginary storage.

Uses separate Float64 arrays for real and imaginary parts so that CUDA
kernels can use atomic Float64 adds directly — no temporary allocation needed.

Shape: (N, N, Nw_layers) for each of data_re, data_im, weights.
"""
struct GPUGrid{T<:AbstractArray{Float64}}
    data_re::T     # Real part of visibility grid (N, N, Nw)
    data_im::T     # Imaginary part of visibility grid (N, N, Nw)
    weights::T     # Sampling density weights (N, N, Nw)
    w_values::Vector{Float64}  # w-value at center of each layer
    
    function GPUGrid(size::Int, w_layers::Int; gpu::Bool=true)
        w_values = zeros(Float64, w_layers)
        
        if gpu && CUDA.functional()
            data_re = CUDA.zeros(Float64, size, size, w_layers)
            data_im = CUDA.zeros(Float64, size, size, w_layers)
            weights = CUDA.zeros(Float64, size, size, w_layers)
            new{typeof(data_re)}(data_re, data_im, weights, w_values)
        else
            data_re = zeros(Float64, size, size, w_layers)
            data_im = zeros(Float64, size, size, w_layers)
            weights = zeros(Float64, size, size, w_layers)
            new{typeof(data_re)}(data_re, data_im, weights, w_values)
        end
    end
end

"""Reset grid to zeros."""
function Base.empty!(grid::GPUGrid)
    fill!(grid.data_re, zero(Float64))
    fill!(grid.data_im, zero(Float64))
    fill!(grid.weights, zero(Float64))
    fill!(grid.w_values, 0.0)
    return grid
end

#==============================================================================#
#                           Image Types                                         #
#==============================================================================#

"""
    GPUImage(size; gpu=true)

GPU-friendly image storage for full Stokes polarization.
All images are 2D arrays of Float64 (Nu x Nv).
"""
struct GPUImage{T<:AbstractArray{Float64}}
    stokes_I::T
    stokes_Q::T
    stokes_U::T
    stokes_V::T
    
    function GPUImage(size::Int; gpu::Bool=true)
        if gpu && CUDA.functional()
            I = CUDA.zeros(Float64, size, size)
            Q = CUDA.zeros(Float64, size, size)
            U = CUDA.zeros(Float64, size, size)
            V = CUDA.zeros(Float64, size, size)
            new{typeof(I)}(I, Q, U, V)
        else
            I = zeros(Float64, size, size)
            Q = zeros(Float64, size, size)
            U = zeros(Float64, size, size)
            V = zeros(Float64, size, size)
            new{typeof(I)}(I, Q, U, V)
        end
    end
end

"""
    image_coordinates(img, config) -> (l_coords, m_coords)

Compute image coordinate arrays (l, m) in radians.
"""
function image_coordinates(img::GPUImage, config::GPUImagerConfig)
    N = size(img.stokes_I, 1)
    cell = config.cell_size
    half = N ÷ 2
    l_coords = [(i - half - 1) * cell for i in 1:N]
    m_coords = [(j - half - 1) * cell for j in 1:N]
    return l_coords, m_coords
end

#==============================================================================#
#                         Gridding Functions                                    #
#==============================================================================#

"""
Compute w-layer index for each visibility.
"""
function w_layer_indices(w_values::AbstractVector, config::GPUImagerConfig)
    Nw = config.w_layers
    
    if Nw == 1
        return ones(Int, length(w_values))
    end
    
    w_min, w_max = extrema(w_values)
    w_range = w_max - w_min
    
    if w_range < 1e-10
        return ones(Int, length(w_values))
    end
    
    layers = floor.(Int, (w_values .- w_min) ./ w_range .* Nw) .+ 1
    return clamp.(layers, 1, Nw)
end

"""
    grid_visibilities!(grid, vis, meta, config) -> grid

Grid visibilities onto UV grid with w-stacking.
Dispatches to GPU or CPU implementation based on array types.
"""
function grid_visibilities!(grid::GPUGrid, vis::GPUVisibilities, 
                           meta::GPUMetadata, config::GPUImagerConfig)
    use_gpu = _is_gpu(vis)
    
    if use_gpu
        gpu_grid_visibilities!(grid, vis, meta, config)
    else
        cpu_grid_visibilities!(grid, vis, meta, config)
    end
    
    return grid
end

"""
CPU implementation of visibility gridding with w-stacking.
"""
function cpu_grid_visibilities!(grid::GPUGrid, vis::GPUVisibilities,
                                meta::GPUMetadata, config::GPUImagerConfig)
    N = size(grid.data_re, 1)  # padded grid size
    Nw = config.w_layers
    cell = config.cell_size
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    
    # Get arrays on CPU
    uvw = meta.uvw isa CUDA.CuArray ? Array(meta.uvw) : meta.uvw
    channels = meta.channels isa CUDA.CuArray ? Array(meta.channels) : meta.channels
    flags = vis.flags isa CUDA.CuArray ? Array(vis.flags) : vis.flags
    
    vis_xx = vis.xx isa CUDA.CuArray ? Array(vis.xx) : vis.xx
    vis_yy = vis.yy isa CUDA.CuArray ? Array(vis.yy) : vis.yy
    vis_xy = vis.xy isa CUDA.CuArray ? Array(vis.xy) : vis.xy
    vis_yx = vis.yx isa CUDA.CuArray ? Array(vis.yx) : vis.yx
    
    c = 299792458.0
    uv_cell = 1.0 / (N * cell)
    center = N ÷ 2 + 1
    
    # Reset grid
    empty!(grid)
    
    # Compute w-layer boundaries (include both +w and -w for Hermitian conjugates)
    w_min = Inf
    w_max = -Inf
    for β in 1:Nf
        λ = c / channels[β]
        for α in 1:Nb
            w_λ = uvw[3, α] / λ
            w_min = min(w_min, w_λ, -w_λ)
            w_max = max(w_max, w_λ, -w_λ)
        end
    end
    if w_min > w_max
        return grid
    end
    w_range = max(w_max - w_min, 1e-10)
    
    # Set layer center w-values
    for w in 1:Nw
        grid.w_values[w] = w_min + (w - 0.5) * w_range / Nw
    end
    
    # Grid each visibility
    support = config.support
    W = Float64(support)
    beta = 2.34 * W
    inv_i0beta = support > 0 ? 1.0 / _besseli0(beta) : 0.0
    
    @inbounds for β in 1:Nf
        λ = c / channels[β]
        
        for α in 1:Nb
            if flags[α, β]
                continue
            end
            
            u = uvw[1, α] / λ
            v = uvw[2, α] / λ
            w = uvw[3, α] / λ
            
            # Stokes I from XX and YY
            V = 0.5 * (vis_xx[α, β] + vis_yy[α, β])
            V_re = real(V)
            V_im = imag(V)
            
            if support > 0
                # Convolutional gridding: scatter to nearby cells with KB kernel
                u_grid = u / uv_cell + center
                v_grid = v / uv_cell + center
                
                # W-layer assignment
                iw = Nw == 1 ? 1 : clamp(floor(Int, (w - w_min) / w_range * Nw) + 1, 1, Nw)
                
                iu_min = max(1, floor(Int, u_grid) - support)
                iu_max = min(N, ceil(Int, u_grid) + support)
                iv_min = max(1, floor(Int, v_grid) - support)
                iv_max = min(N, ceil(Int, v_grid) + support)
                
                for jv in iv_min:iv_max
                    dv = jv - v_grid
                    kv = _kb_value(dv, W, beta, inv_i0beta)
                    kv == 0.0 && continue
                    for ju in iu_min:iu_max
                        du = ju - u_grid
                        kw = _kb_value(du, W, beta, inv_i0beta) * kv
                        kw == 0.0 && continue
                        grid.data_re[ju, jv, iw] += V_re * kw
                        grid.data_im[ju, jv, iw] += V_im * kw
                        grid.weights[ju, jv, iw] += kw
                    end
                end
                
                # Hermitian conjugate
                u_conj_grid = -u / uv_cell + center
                v_conj_grid = -v / uv_cell + center
                neg_w = -w
                iw_conj = Nw == 1 ? 1 : clamp(floor(Int, (neg_w - w_min) / w_range * Nw) + 1, 1, Nw)
                
                iu_min_c = max(1, floor(Int, u_conj_grid) - support)
                iu_max_c = min(N, ceil(Int, u_conj_grid) + support)
                iv_min_c = max(1, floor(Int, v_conj_grid) - support)
                iv_max_c = min(N, ceil(Int, v_conj_grid) + support)
                
                for jv in iv_min_c:iv_max_c
                    dv = jv - v_conj_grid
                    kv = _kb_value(dv, W, beta, inv_i0beta)
                    kv == 0.0 && continue
                    for ju in iu_min_c:iu_max_c
                        du = ju - u_conj_grid
                        kw = _kb_value(du, W, beta, inv_i0beta) * kv
                        kw == 0.0 && continue
                        grid.data_re[ju, jv, iw_conj] += V_re * kw
                        grid.data_im[ju, jv, iw_conj] -= V_im * kw  # conj
                        grid.weights[ju, jv, iw_conj] += kw
                    end
                end
            else
                # Nearest-neighbor gridding
                iu = round(Int, u / uv_cell) + center
                iv = round(Int, v / uv_cell) + center
                
                if iu < 1 || iu > N || iv < 1 || iv > N
                    continue
                end
                
                iw = Nw == 1 ? 1 : clamp(floor(Int, (w - w_min) / w_range * Nw) + 1, 1, Nw)
                
                grid.data_re[iu, iv, iw] += V_re
                grid.data_im[iu, iv, iw] += V_im
                grid.weights[iu, iv, iw] += 1.0
                
                # Hermitian conjugate
                iu_conj = N - iu + 2
                iv_conj = N - iv + 2
                if iu_conj >= 1 && iu_conj <= N && iv_conj >= 1 && iv_conj <= N
                    neg_w = -w
                    iw_conj = Nw == 1 ? 1 : clamp(floor(Int, (neg_w - w_min) / w_range * Nw) + 1, 1, Nw)
                    grid.data_re[iu_conj, iv_conj, iw_conj] += V_re
                    grid.data_im[iu_conj, iv_conj, iw_conj] -= V_im  # conj
                    grid.weights[iu_conj, iv_conj, iw_conj] += 1.0
                end
            end
        end
    end
    
    apply_weighting!(grid, config)
    return grid
end

"""
GPU implementation of visibility gridding using CUDA kernel.

Launches `grid_nn_kernel!` which writes directly to the grid's split re/im
arrays via atomic Float64 adds — no temporary 3D allocation needed.
"""
function gpu_grid_visibilities!(grid::GPUGrid, vis::GPUVisibilities,
                                meta::GPUMetadata, config::GPUImagerConfig)
    N = size(grid.data_re, 1)  # padded grid size
    Nw = config.w_layers
    cell = config.cell_size
    Nb = Nbase(vis)
    Nf = Nfreq(vis)
    c = 299792458.0
    
    uv_cell = 1.0 / (N * cell)
    center = Int32(N ÷ 2 + 1)
    
    empty!(grid)
    
    # Compute w-range on CPU (small data transfer)
    uvw_cpu = Array(meta.uvw)
    channels_cpu = Array(meta.channels)
    
    w_min = Inf
    w_max = -Inf
    for β in 1:Nf
        λ = c / channels_cpu[β]
        for α in 1:Nb
            w_λ = uvw_cpu[3, α] / λ
            w_min = min(w_min, w_λ, -w_λ)
            w_max = max(w_max, w_λ, -w_λ)
        end
    end
    if w_min > w_max
        return grid
    end
    w_range = max(w_max - w_min, 1e-10)
    
    for w in 1:Nw
        grid.w_values[w] = w_min + (w - 0.5) * w_range / Nw
    end
    
    # Extract real/imag parts of visibilities on GPU (small: Nbase × Nfreq)
    vis_xx_re = real.(vis.xx)
    vis_xx_im = imag.(vis.xx)
    vis_yy_re = real.(vis.yy)
    vis_yy_im = imag.(vis.yy)
    
    # Launch gridding kernel — writes DIRECTLY to grid.data_re/im/weights
    # No temporary N×N×Nw arrays needed!
    total_items = Nb * Nf
    threads = min(256, total_items)
    blocks = cld(total_items, threads)
    
    support = Int32(config.support)
    if support > 0
        @cuda blocks=blocks threads=threads grid_convolve_kernel!(
            grid.data_re, grid.data_im, grid.weights,
            vis_xx_re, vis_xx_im,
            vis_yy_re, vis_yy_im,
            vis.flags,
            meta.uvw, meta.channels,
            Float64(w_min), Float64(w_range), Int32(Nw),
            Int32(N), Float64(uv_cell), center,
            support,
            Int32(Nb), Int32(Nf)
        )
    else
        @cuda blocks=blocks threads=threads grid_nn_kernel!(
            grid.data_re, grid.data_im, grid.weights,
            vis_xx_re, vis_xx_im,
            vis_yy_re, vis_yy_im,
            vis.flags,
            meta.uvw, meta.channels,
            Float64(w_min), Float64(w_range), Int32(Nw),
            Int32(N), Float64(uv_cell), center,
            Int32(Nb), Int32(Nf)
        )
    end
    CUDA.synchronize()
    
    # Free vis re/im temporaries
    CUDA.unsafe_free!(vis_xx_re)
    CUDA.unsafe_free!(vis_xx_im)
    CUDA.unsafe_free!(vis_yy_re)
    CUDA.unsafe_free!(vis_yy_im)
    
    apply_weighting!(grid, config)
    return grid
end

"""
Apply visibility weighting (natural, uniform, or Briggs).

Weighting schemes (matching wsclean):
- Natural: weight = 1 per sample. Best point-source sensitivity.
- Uniform: divide each cell by its sample count. Higher resolution but higher noise.
- Briggs: interpolates between natural (robust=+2) and uniform (robust=-2).
"""
function apply_weighting!(grid::GPUGrid, config::GPUImagerConfig)
    if config.weighting == :natural
        return grid
        
    elseif config.weighting == :uniform
        safe_w = max.(grid.weights, one(Float64))
        grid.data_re ./= safe_w
        grid.data_im ./= safe_w
        grid.weights .= sign.(grid.weights)
        
    elseif config.weighting == :briggs
        robust = config.robust
        total_weight = sum(grid.weights)
        sum_w2 = sum(grid.weights .^ 2)
        if sum_w2 > 0 && total_weight > 0
            f2 = (5.0 * 10.0^(-robust))^2 / (sum_w2 / total_weight)
        else
            f2 = 1.0
        end
        briggs_w = 1.0 ./ (1.0 .+ f2 .* grid.weights)
        grid.data_re .*= briggs_w
        grid.data_im .*= briggs_w
        grid.weights .*= briggs_w
    end
    
    return grid
end

#==============================================================================#
#                        W-Term Correction                                      #
#==============================================================================#

"""
    w_correction_phase(l, m, w) -> ComplexF64

Compute w-correction phase for a point (l, m) at w-value w (in wavelengths).

The w-correction applies the phase: exp(2πi w (√(1-l²-m²) - 1))
"""
function w_correction_phase(l::Float64, m::Float64, w::Float64)
    r2 = l^2 + m^2
    if r2 >= 1.0
        return zero(ComplexF64)
    end
    n = sqrt(1.0 - r2)
    return exp(2π * im * w * (n - 1.0))
end

"""
    combine_w_layers(grid, config) -> Matrix{ComplexF64}

Combine w-layers into single image by applying w-correction in image domain (CPU).
"""
function combine_w_layers(grid::GPUGrid, config::GPUImagerConfig)
    N = size(grid.data_re, 1)  # padded grid size
    Nw = size(grid.data_re, 3)
    cell = config.cell_size
    
    combined = zeros(ComplexF64, N, N)
    half = N ÷ 2
    
    for iw in 1:Nw
        layer_data = complex.(grid.data_re[:, :, iw], grid.data_im[:, :, iw])
        layer_fft = fftshift(ifft(ifftshift(layer_data))) .* N^2
        w_layer = grid.w_values[iw]
        
        for iv in 1:N
            m = (iv - half - 1) * cell
            for iu in 1:N
                l = (iu - half - 1) * cell
                correction = w_correction_phase(l, m, w_layer)
                combined[iu, iv] += layer_fft[iu, iv] * correction
            end
        end
    end
    
    return combined
end

#==============================================================================#
#                        Image Formation                                        #
#==============================================================================#

"""
    grid_to_image!(img, grid, config) -> img

Convert gridded visibilities to image via FFT with w-correction.
Dispatches to GPU or CPU implementation.
"""
function grid_to_image!(img::GPUImage, grid::GPUGrid, config::GPUImagerConfig)
    use_gpu = grid.data_re isa CUDA.CuArray
    
    if use_gpu
        gpu_grid_to_image!(img, grid, config)
    else
        cpu_grid_to_image!(img, grid, config)
    end
    
    return img
end

"""
    gridding_correction(N, support) -> Matrix{Float64}

Compute gridding correction image to compensate for the gridding kernel taper.

- support=0 (NN): correction for sinc taper from sub-cell quantization.
  Max correction ≈ π/2 per axis (~2.5× in 2D corners). Well-conditioned.
- support>0 (convolution): DFT of the truncated Gaussian kernel, with cap
  to avoid extreme amplification at image edges.

Returns a 2D array of correction factors (=1 at center, ≥1 elsewhere).
"""
function gridding_correction(N::Int, support::Int)
    center = N ÷ 2 + 1
    corr_1d = Vector{Float64}(undef, N)
    
    if support <= 0
        # NN gridding: sinc correction for sub-cell quantization error.
        # Averaging over random sub-cell offsets attenuates by sinc(k/N).
        for i in 1:N
            x = (i - center) / N  # fractional pixel offset (-0.5 to 0.5)
            if abs(x) < 1e-10
                corr_1d[i] = 1.0
            else
                corr_1d[i] = (π * x) / sin(π * x)  # 1/sinc(x)
            end
        end
    else
        # Convolution gridding: DFT of the actual truncated Kaiser-Bessel kernel.
        W = Float64(support)
        beta = 2.34 * W
        inv_i0beta = 1.0 / _besseli0(beta)
        kernel = zeros(Float64, N)
        for n in -support:support
            kernel[mod(n, N) + 1] = _kb_value(Float64(n), W, beta, inv_i0beta)
        end
        taper = abs.(fftshift(fft(kernel)))
        taper ./= taper[center]  # normalize so center = 1
        for i in 1:N
            corr_1d[i] = 1.0 / max(taper[i], 0.01)  # cap at 100×
        end
    end
    
    # 2D separable: correction[i,j] = corr_1d[i] * corr_1d[j]
    return corr_1d * corr_1d'
end

"""CPU implementation of grid to image conversion."""
function cpu_grid_to_image!(img::GPUImage, grid::GPUGrid, config::GPUImagerConfig)
    N_grid = size(grid.data_re, 1)  # padded grid size
    N_img = config.image_size
    Nw = config.w_layers
    
    if Nw == 1
        layer_data = complex.(grid.data_re[:, :, 1], grid.data_im[:, :, 1])
        dirty = real.(fftshift(ifft(ifftshift(layer_data)))) .* N_grid^2
    else
        combined = combine_w_layers(grid, config)
        dirty = real.(combined)
    end
    
    total_weight = sum(grid.weights)
    if total_weight > 0
        dirty ./= total_weight
    end
    
    # Apply gridding correction at padded size
    dirty .*= gridding_correction(N_grid, config.support)
    
    # Crop center N_img × N_img from padded image
    offset = (N_grid - N_img) ÷ 2
    copyto!(img.stokes_I, @view dirty[offset+1:offset+N_img, offset+1:offset+N_img])
    fill!(img.stokes_Q, 0.0)
    fill!(img.stokes_U, 0.0)
    fill!(img.stokes_V, 0.0)
    
    return img
end

"""
GPU implementation of grid to image conversion.

Pre-allocates all working buffers and reuses them across w-layers.
Uses in-place cuFFT (plan_ifft!) and fftshift_2d! to avoid per-iteration
allocations — the main speedup over the previous implementation.

Memory: 2 × N² ComplexF64 + 4 × N² Float64 ≈ N² × 64 bytes of working space.
For N=4096: ~1 GiB regardless of number of w-layers.
"""
function gpu_grid_to_image!(img::GPUImage, grid::GPUGrid, config::GPUImagerConfig)
    N_grid = size(grid.data_re, 1)  # padded grid size
    N_img = config.image_size
    Nw = config.w_layers
    cell = config.cell_size
    center = Int32(N_grid ÷ 2 + 1)
    
    # Pre-allocate FFT working buffers at padded size
    buf_a = CUDA.zeros(ComplexF64, N_grid, N_grid)
    buf_b = CUDA.zeros(ComplexF64, N_grid, N_grid)
    plan = plan_ifft!(buf_b)
    
    # Padded real-image accumulator
    padded = CUDA.zeros(Float64, N_grid, N_grid)
    
    if Nw == 1
        buf_a .= complex.(@view(grid.data_re[:, :, 1]), @view(grid.data_im[:, :, 1]))
        fftshift_2d!(buf_b, buf_a)
        plan * buf_b
        fftshift_2d!(buf_a, buf_b)
        buf_a .*= Float64(N_grid^2)
        padded .= real.(buf_a)
    else
        # W-stacking: cuFFT each layer + GPU w-correction kernel
        layer_re = CUDA.zeros(Float64, N_grid, N_grid)
        layer_im = CUDA.zeros(Float64, N_grid, N_grid)
        padded_im = CUDA.zeros(Float64, N_grid, N_grid)
        
        total_pixels = N_grid * N_grid
        threads = min(256, total_pixels)
        blocks = cld(total_pixels, threads)
        
        for iw in 1:Nw
            buf_a .= complex.(@view(grid.data_re[:, :, iw]), @view(grid.data_im[:, :, iw]))
            fftshift_2d!(buf_b, buf_a)
            plan * buf_b
            fftshift_2d!(buf_a, buf_b)
            buf_a .*= Float64(N_grid^2)
            
            layer_re .= real.(buf_a)
            layer_im .= imag.(buf_a)
            
            @cuda blocks=blocks threads=threads w_correction_kernel!(
                padded, padded_im,
                layer_re, layer_im,
                Float64(grid.w_values[iw]),
                Int32(N_grid), Float64(cell),
                center
            )
            CUDA.synchronize()
        end
        
        CUDA.unsafe_free!(padded_im)
        CUDA.unsafe_free!(layer_re)
        CUDA.unsafe_free!(layer_im)
    end
    
    # Normalize by total weight
    total_weight = sum(grid.weights)
    if total_weight > 0
        padded ./= total_weight
    end
    
    # Apply gridding correction at padded size
    gc = CuArray(gridding_correction(N_grid, config.support))
    padded .*= gc
    CUDA.unsafe_free!(gc)
    
    # Crop center N_img × N_img from padded image
    offset = (N_grid - N_img) ÷ 2
    img.stokes_I .= @view padded[offset+1:offset+N_img, offset+1:offset+N_img]
    CUDA.unsafe_free!(padded)
    
    fill!(img.stokes_Q, 0.0)
    fill!(img.stokes_U, 0.0)
    fill!(img.stokes_V, 0.0)
    
    CUDA.unsafe_free!(buf_a)
    CUDA.unsafe_free!(buf_b)
    
    return img
end

#==============================================================================#
#                        High-Level API                                         #
#==============================================================================#

"""
    make_image(vis, meta, config) -> GPUImage

Create a dirty image from visibilities.

# Arguments
- `vis::GPUVisibilities`: Input visibilities
- `meta::GPUMetadata`: Observation metadata
- `config::GPUImagerConfig`: Imaging configuration

# Returns
- `GPUImage`: Dirty image (Stokes I, Q, U, V)
"""
function make_image(vis::GPUVisibilities, meta::GPUMetadata, config::GPUImagerConfig)
    use_gpu = _is_gpu(vis)
    N_pad = _padded_size(config)
    
    grid = GPUGrid(N_pad, config.w_layers, gpu=use_gpu)
    img = GPUImage(config.image_size, gpu=use_gpu)
    
    grid_visibilities!(grid, vis, meta, config)
    grid_to_image!(img, grid, config)
    
    return img
end

"""
    peel_and_image(vis, meta, sources, config; kwargs...) -> GPUImage

Peel sources and create image without intermediate MS files.

This is the main workflow function that keeps all data on GPU:
1. Peel specified sources from visibilities
2. Grid residual visibilities
3. Apply w-correction
4. FFT to create dirty image

# Arguments
- `vis::GPUVisibilities`: Input visibilities (modified in place)
- `meta::GPUMetadata`: Observation metadata  
- `sources::Vector{<:AbstractGPUPeelingSource}`: Sources to peel
- `config::GPUImagerConfig`: Imaging configuration

# Keyword Arguments
- `peeliter::Int=3`: Number of peeling iterations
- `maxiter::Int=20`: Max stefcal iterations per source
- `tolerance::Float64=1e-3`: Stefcal convergence tolerance
- `minuvw::Float64=0.0`: Minimum baseline length in wavelengths
- `phase_center_ra::Float64=0.0`: Phase center RA (radians)
- `phase_center_dec::Float64=0.0`: Phase center Dec (radians)  
- `lst::Float64=0.0`: Local sidereal time (radians)

# Returns
- `GPUImage`: Dirty image of residuals after peeling
"""
function peel_and_image(vis::GPUVisibilities, meta::GPUMetadata,
                        sources::Vector{T}, config::GPUImagerConfig;
                        peeliter::Int=3, maxiter::Int=20, tolerance::Float64=1e-3,
                        minuvw::Float64=0.0,
                        phase_center_ra::Float64=0.0, phase_center_dec::Float64=0.0,
                        lst::Float64=0.0) where {T}
    
    # Step 1: Peel sources (keeps data on GPU)
    log_step("Peeling $(length(sources)) sources...")
    calibrations = peel_gpu!(vis, meta, sources;
                             peeliter=peeliter, maxiter=maxiter, tolerance=tolerance,
                             minuvw=minuvw,
                             phase_center_ra=phase_center_ra, phase_center_dec=phase_center_dec,
                             lst=lst)
    
    # Step 2: Image the residuals (still on GPU)
    log_step("Imaging residuals...")
    img = make_image(vis, meta, config)
    
    log_success("Peel-and-image complete")
    return img
end

end # module GPUImager
