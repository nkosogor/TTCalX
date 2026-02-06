# GPU Visibility Generation Kernels
# GPU-accelerated model visibility computation

using CUDA

#==============================================================================#
#                       Fringe Computation Kernels                             #
#==============================================================================#

"""
Kernel to compute geometric delays and fringes for all antennas.
Output: fringes array (Nant, Nfreq) of complex values.

NOTE: This kernel uses antenna positions (ITRF) directly with direction cosines.
This is a simplified approach. For phase-referenced observations where UVW 
coordinates are available, use the compute_baseline_fringes_kernel! instead.
"""
function compute_fringes_kernel!(
    # Output: fringes (Nant, Nfreq)
    fringes,
    # Antenna positions (3, Nant)
    antenna_positions,
    # Source direction (l, m, n) relative to phase center
    source_l, source_m, source_n,
    # Frequency channels (Nfreq,)
    channels,
    # Dimensions
    Nant, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nant * Nfreq
        ant = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        
        # Antenna position
        x = antenna_positions[1, ant]
        y = antenna_positions[2, ant]
        z = antenna_positions[3, ant]
        
        # Geometric delay
        delay = (x*source_l + y*source_m + z*source_n) / GPU_C
        
        # Frequency
        ν = channels[β]
        
        # Fringe
        ϕ = 2π * ν * delay
        fringes[ant, β] = exp(1im * ϕ)
    end
    
    return nothing
end


#==============================================================================#
#                    UVW-Based Fringe Computation                              #
#==============================================================================#

"""
Kernel to compute baseline fringes using UVW coordinates.
The UVW coordinates from the MS are in the phase center direction.
For a source at direction (l, m, n) relative to the phase center:
    phase = 2π * (u*l + v*m + w*(n-1)) / λ

This is the correct approach for phase-referenced observations.
"""
function compute_baseline_fringes_kernel!(
    # Output: fringes per baseline (Nbase, Nfreq)
    baseline_fringes,
    # UVW coordinates (3, Nbase) in meters
    uvw,
    # Source direction cosines relative to phase center
    source_l, source_m, source_n,
    # Frequency channels (Nfreq,)
    channels,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        # UVW for this baseline (in meters)
        u = uvw[1, α]
        v = uvw[2, α]
        w = uvw[3, α]
        
        # Frequency and wavelength
        ν = channels[β]
        λ = GPU_C / ν
        
        # Convert UVW to wavelengths
        uλ = u / λ
        vλ = v / λ
        wλ = w / λ
        
        # Phase: 2π * (u*l + v*m + w*(n-1))
        # The (n-1) term comes from the visibility equation
        ϕ = 2π * (uλ * source_l + vλ * source_m + wλ * (source_n - 1.0))
        
        baseline_fringes[α, β] = exp(1im * ϕ)
    end
    
    return nothing
end


#==============================================================================#
#                    Point Source Visibility Kernel                            #
#==============================================================================#

"""
Kernel to add point source contribution to visibilities using baseline fringes.
V[α,β] += flux * fringe[α,β]
"""
function genvis_point_uvw_kernel!(
    # Output visibilities (Nbase, Nfreq) - accumulated
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Baseline fringes (Nbase, Nfreq)
    baseline_fringes,
    # Flux (4 components for full Jones, indexed by frequency)
    flux_xx, flux_xy, flux_yx, flux_yy,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        # Get fringe for this baseline
        fringe = baseline_fringes[α, β]
        
        # Get flux for this frequency
        fxx = flux_xx[β]
        fxy = flux_xy[β]
        fyx = flux_yx[β]
        fyy = flux_yy[β]
        
        # Accumulate: V += flux * fringe
        vis_xx[α, β] += fxx * fringe
        vis_xy[α, β] += fxy * fringe
        vis_yx[α, β] += fyx * fringe
        vis_yy[α, β] += fyy * fringe
    end
    
    return nothing
end

"""
Kernel to add point source contribution to visibilities.
V[α,β] += flux * fringe[ant1] * conj(fringe[ant2])

NOTE: This uses the old antenna-based fringe approach. For phase-referenced
observations, use genvis_point_uvw_kernel! instead.
"""
function genvis_point_kernel!(
    # Output visibilities (Nbase, Nfreq) - accumulated
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Fringes (Nant, Nfreq)
    fringes,
    # Flux (4 components for full Jones, indexed by frequency)
    flux_xx, flux_xy, flux_yx, flux_yy,
    # Baseline indices (2, Nbase)
    baselines,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Fringe: exp(iφ₁) * exp(-iφ₂) = exp(i(φ₁-φ₂))
        fringe = fringes[ant1, β] * conj(fringes[ant2, β])
        
        # Get flux for this frequency
        fxx = flux_xx[β]
        fxy = flux_xy[β]
        fyx = flux_yx[β]
        fyy = flux_yy[β]
        
        # Accumulate: V += flux * fringe
        vis_xx[α, β] += fxx * fringe
        vis_xy[α, β] += fxy * fringe
        vis_yx[α, β] += fyx * fringe
        vis_yy[α, β] += fyy * fringe
    end
    
    return nothing
end


#==============================================================================#
#                   Gaussian Source Visibility Kernel                          #
#==============================================================================#

"""
Kernel for Gaussian source with baseline coherency attenuation.
The visibility is attenuated by exp(-width * baseline_projection²)
"""
function genvis_gaussian_kernel!(
    # Output visibilities (Nbase, Nfreq) - accumulated
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Fringes (Nant, Nfreq)
    fringes,
    # Flux (4 components, indexed by frequency)
    flux_xx, flux_xy, flux_yx, flux_yy,
    # Baseline indices (2, Nbase)
    baselines,
    # Antenna positions (3, Nant)
    antenna_positions,
    # Gaussian parameters
    major_axis_x, major_axis_y, major_axis_z,  # Major axis unit vector
    minor_axis_x, minor_axis_y, minor_axis_z,  # Minor axis unit vector
    major_width,  # π² * sin(FWHM)² / (4ln2)
    minor_width,
    # Frequency channels (Nfreq,)
    channels,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Baseline vector
        x1 = antenna_positions[1, ant1]
        y1 = antenna_positions[2, ant1]
        z1 = antenna_positions[3, ant1]
        x2 = antenna_positions[1, ant2]
        y2 = antenna_positions[2, ant2]
        z2 = antenna_positions[3, ant2]
        
        # Convert to wavelengths
        ν = channels[β]
        λ = GPU_C / ν
        u = (x1 - x2) / λ
        v = (y1 - y2) / λ
        w = (z1 - z2) / λ
        
        # Project baseline onto major/minor axes
        major_proj = u*major_axis_x + v*major_axis_y + w*major_axis_z
        minor_proj = u*minor_axis_x + v*minor_axis_y + w*minor_axis_z
        
        # Gaussian attenuation
        coherency = exp(-major_width*major_proj^2 - minor_width*minor_proj^2)
        
        # Fringe
        fringe = fringes[ant1, β] * conj(fringes[ant2, β])
        
        # Flux
        fxx = flux_xx[β]
        fxy = flux_xy[β]
        fyx = flux_yx[β]
        fyy = flux_yy[β]
        
        # Accumulate: V += flux * fringe * coherency
        factor = fringe * coherency
        vis_xx[α, β] += fxx * factor
        vis_xy[α, β] += fxy * factor
        vis_yx[α, β] += fyx * factor
        vis_yy[α, β] += fyy * factor
    end
    
    return nothing
end


#==============================================================================#
#                        Disk Source Visibility Kernel                         #
#==============================================================================#

"""
Kernel for disk (uniform disk) source.
Uses Bessel function for coherency: J₁(2πθb)/(πθb)
"""
function genvis_disk_kernel!(
    # Output visibilities (Nbase, Nfreq) - accumulated
    vis_xx, vis_xy, vis_yx, vis_yy,
    # Fringes (Nant, Nfreq)
    fringes,
    # Flux (4 components, indexed by frequency)
    flux_xx, flux_xy, flux_yx, flux_yy,
    # Baseline indices (2, Nbase)
    baselines,
    # Antenna positions (3, Nant)
    antenna_positions,
    # Disk parameters
    radius,  # Angular radius in radians
    # Frequency channels (Nfreq,)
    channels,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        ant1 = baselines[1, α]
        ant2 = baselines[2, α]
        
        # Baseline vector in wavelengths
        x1 = antenna_positions[1, ant1]
        y1 = antenna_positions[2, ant1]
        z1 = antenna_positions[3, ant1]
        x2 = antenna_positions[1, ant2]
        y2 = antenna_positions[2, ant2]
        z2 = antenna_positions[3, ant2]
        
        ν = channels[β]
        λ = GPU_C / ν
        u = (x1 - x2) / λ
        v = (y1 - y2) / λ
        w = (z1 - z2) / λ
        
        # Baseline length
        b = sqrt(u^2 + v^2 + w^2)
        
        # Disk coherency: J₁(2πθb)/(πθb)
        θb = radius * b
        if θb < eps(Float64)
            coherency = 1.0
        else
            # Approximate Bessel J₁ for GPU
            x = 2π * θb
            if abs(x) < 4.0
                # Series expansion for small x
                coherency = 0.5 * (1.0 - x^2/8.0 + x^4/192.0 - x^6/9216.0)
            else
                # Asymptotic form for large x (simplified)
                coherency = sqrt(2.0/(π*x)) * cos(x - 3π/4) / (π*θb)
            end
        end
        
        # Fringe
        fringe = fringes[ant1, β] * conj(fringes[ant2, β])
        
        # Flux
        fxx = flux_xx[β]
        fxy = flux_xy[β]
        fyx = flux_yx[β]
        fyy = flux_yy[β]
        
        # Accumulate
        factor = fringe * coherency
        vis_xx[α, β] += fxx * factor
        vis_xy[α, β] += fxy * factor
        vis_yx[α, β] += fyx * factor
        vis_yy[α, β] += fyy * factor
    end
    
    return nothing
end


#==============================================================================#
#                        Subtract/Add Source Kernels                           #
#==============================================================================#

"""
Kernel to subtract model from data: data -= model
"""
function subsrc_kernel!(
    # Data to modify (Nbase, Nfreq)
    data_xx, data_xy, data_yx, data_yy,
    # Model to subtract (Nbase, Nfreq)
    model_xx, model_xy, model_yx, model_yy,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        data_xx[α, β] -= model_xx[α, β]
        data_xy[α, β] -= model_xy[α, β]
        data_yx[α, β] -= model_yx[α, β]
        data_yy[α, β] -= model_yy[α, β]
    end
    
    return nothing
end

"""
Kernel to add model to data: data += model
"""
function putsrc_kernel!(
    # Data to modify (Nbase, Nfreq)
    data_xx, data_xy, data_yx, data_yy,
    # Model to add (Nbase, Nfreq)
    model_xx, model_xy, model_yx, model_yy,
    # Dimensions
    Nbase, Nfreq
)
    idx = thread_index_1d()
    
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        
        data_xx[α, β] += model_xx[α, β]
        data_xy[α, β] += model_xy[α, β]
        data_yx[α, β] += model_yx[α, β]
        data_yy[α, β] += model_yy[α, β]
    end
    
    return nothing
end


#==============================================================================#
#                        High-Level Functions                                  #
#==============================================================================#

"""
GPU-accelerated source subtraction.
"""
function gpu_subsrc!(data::GPUVisibilities, model::GPUVisibilities)
    Nb = Nbase(data)
    Nf = Nfreq(data)
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    @cuda blocks=blocks threads=threads subsrc_kernel!(
        data.xx, data.xy, data.yx, data.yy,
        model.xx, model.xy, model.yx, model.yy,
        Nb, Nf
    )
    
    CUDA.synchronize()
    data
end

"""
GPU-accelerated source addition.
"""
function gpu_putsrc!(data::GPUVisibilities, model::GPUVisibilities)
    Nb = Nbase(data)
    Nf = Nfreq(data)
    
    blocks, threads = kernel_config_1d(Nb * Nf)
    
    @cuda blocks=blocks threads=threads putsrc_kernel!(
        data.xx, data.xy, data.yx, data.yy,
        model.xx, model.xy, model.yx, model.yy,
        Nb, Nf
    )
    
    CUDA.synchronize()
    data
end


"""
    gpu_genvis!(vis, meta, source_params; source_type=:point)

Generate model visibilities on GPU for a single source.

# Arguments
- `vis`: GPUVisibilities to accumulate into
- `meta`: GPUMetadata with antenna positions, baselines, channels
- `source_params`: Dictionary with source parameters
  - `:direction` => (l, m, n) relative to phase center
  - `:flux` => (flux_xx, flux_xy, flux_yx, flux_yy) arrays per frequency
  - For Gaussian: `:major_axis`, `:minor_axis`, `:major_width`, `:minor_width`
  - For Disk: `:radius`
- `source_type`: :point, :gaussian, or :disk
"""
function gpu_genvis!(vis::GPUVisibilities, meta::GPUMetadata, source_params::Dict;
                     source_type::Symbol=:point)
    Na = Nant(meta)
    Nb = Nbase(meta)
    Nf = Nfreq(meta)
    
    # Get source direction
    l, m, n = source_params[:direction]
    
    # Allocate fringes array
    fringes = CUDA.zeros(ComplexF64, Na, Nf)
    
    # Compute fringes for all antennas and frequencies
    blocks1, threads1 = kernel_config_1d(Na * Nf)
    @cuda blocks=blocks1 threads=threads1 compute_fringes_kernel!(
        fringes,
        meta.antenna_positions,
        l, m, n,
        meta.channels,
        Na, Nf
    )
    
    # Get flux arrays
    flux_xx, flux_xy, flux_yx, flux_yy = source_params[:flux]
    
    # Generate visibilities based on source type
    blocks2, threads2 = kernel_config_1d(Nb * Nf)
    
    if source_type == :point
        @cuda blocks=blocks2 threads=threads2 genvis_point_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            fringes,
            flux_xx, flux_xy, flux_yx, flux_yy,
            meta.baselines,
            Nb, Nf
        )
        
    elseif source_type == :gaussian
        major_axis = source_params[:major_axis]
        minor_axis = source_params[:minor_axis]
        major_width = source_params[:major_width]
        minor_width = source_params[:minor_width]
        
        @cuda blocks=blocks2 threads=threads2 genvis_gaussian_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            fringes,
            flux_xx, flux_xy, flux_yx, flux_yy,
            meta.baselines,
            meta.antenna_positions,
            major_axis[1], major_axis[2], major_axis[3],
            minor_axis[1], minor_axis[2], minor_axis[3],
            major_width, minor_width,
            meta.channels,
            Nb, Nf
        )
        
    elseif source_type == :disk
        radius = source_params[:radius]
        
        @cuda blocks=blocks2 threads=threads2 genvis_disk_kernel!(
            vis.xx, vis.xy, vis.yx, vis.yy,
            fringes,
            flux_xx, flux_xy, flux_yx, flux_yy,
            meta.baselines,
            meta.antenna_positions,
            radius,
            meta.channels,
            Nb, Nf
        )
    else
        error("Unknown source type: $source_type")
    end
    
    CUDA.synchronize()
    vis
end
