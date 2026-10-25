#==============================================================================#
#                    Sky + RFI peeling of one dataset in one call               #
#==============================================================================#
#
# The building blocks of the README's "Sky and RFI peeling" section put
# together: data flags on every model, exact sky model, near-field RFI model,
# solve_gains! (RK4 by default) inside guarded_zest!, the RFI emitters peeled
# after the sky sources or jointly with them.  bin/zest_sky_rfi.jl runs
# zest_sky_rfi_ms! on MS files.

"""
    source_elevation_deg(source, meta) -> Float64

Elevation (deg) of `source` above the plane normal to the phase centre (the
local horizon for zenith-phased OVRO-LWA snapshots); NaN for RFI emitters.
"""
function source_elevation_deg(source::GPUSource, meta::GPUMetadata)
    is_rfi(source) && return NaN
    lmn = source_direction_lmn(source, meta.phase_center_ra, meta.phase_center_dec, 0.0)
    rad2deg(asin(clamp(lmn[3], -1.0, 1.0)))
end

"""
    zest_sky_rfi!(vis, meta, mjd, sky, rfi=GPUSource[]; kwargs...) -> NamedTuple

Peel the `sky` sources and then the `rfi` emitters from `vis` (GPU or host
arrays).  `meta` should already carry geometric UVW (`geometric_uvw!`).

Keywords:
- `rfi_mode = :after` (separate pass after the sky) or `:joint` (one more direction)
- `solver = :rk4` or `:damped`; `maxiter = 5`, `tolerance = 1e-4`, `minuvw = 5.0`
- `peeliter = 3` peel passes; `max_gain = 100.0` rejects solutions above it
- `min_elevation = 0.0` (deg): sky sources below it are not peeled
- `diagonal = false`: `true` peels (diagonal Jones) instead of zesting (full Jones)
- `reg = nothing` or `(max_offset_deg, taper_deg[, lambda])` for the Tukey phase prior
- `flag_zeros = true`: flag exactly-zero samples first (`flag_zero_visibilities!`)
- `report = nothing` or `(name, iteration; action, kw...) -> ...`, see `guarded_zest!`

Returns `(peeled, skipped, flagged_zeros)`: names of the subtracted sources,
names of the sky sources below `min_elevation`, newly flagged zero samples.
"""
function zest_sky_rfi!(vis::GPUVisibilities, meta::GPUMetadata, mjd::Float64,
                       sky::AbstractVector, rfi::AbstractVector=GPUSource[];
                       rfi_mode::Symbol=:after, solver::Symbol=:rk4, maxiter::Integer=5,
                       tolerance::Real=1e-4, minuvw::Real=5.0, peeliter::Integer=3,
                       max_gain::Real=100.0, min_elevation::Real=0.0, diagonal::Bool=false,
                       reg=nothing, flag_zeros::Bool=true, report=nothing)
    rfi_mode in (:after, :joint) || throw(ArgumentError("rfi_mode must be :after or :joint"))
    gpu = _is_gpu(vis.xx)
    nzero = flag_zeros ? flag_zero_visibilities!(vis) : 0
    prior = reg === nothing ? nothing :
        phase_regularization(meta, reg[1], reg[2]; lambda=length(reg) > 2 ? reg[3] : 1.0, gpu=gpu)
    solve = (cal, v, coh) -> begin
        converged, niters, _ = solve_gains!(cal, v, coh, meta; solver=solver, maxiter=maxiter,
                                            tolerance=tolerance, minuvw=minuvw, reg=prior)
        converged, niters
    end

    up = [source_elevation_deg(s, meta) >= min_elevation for s in sky]
    passes = rfi_mode == :joint ? [vcat(sky[up], rfi)] : [sky[up], rfi]
    wrap = diagonal ? GPUPeelingSource : GPUZestingSource
    peeled = String[]
    for sources in passes
        isempty(sources) && continue
        cals = [calibration_type(wrap(s), meta.Nant, meta.Nfreq; gpu=gpu) for s in sources]
        cohs = map(sources) do s
            c = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=gpu)
            if is_rfi(s)
                gpu_genvis!(c, meta, s, meta.phase_center_ra, meta.phase_center_dec, 0.0)
            else
                genvis_exact!(c, meta, s, mjd)
            end
            c.flags .= vis.flags
            c
        end
        rep = report === nothing ? nothing :
            (i, it; kw...) -> report(get_name(sources[i]), it; kw...)
        sub = guarded_zest!(vis, meta, cohs, cals, solve; peeliter=peeliter,
                            max_gain=max_gain, report=rep)
        append!(peeled, get_name.(sources[sub]))
    end
    (peeled=peeled, skipped=get_name.(sky[.!up]), flagged_zeros=nzero)
end

"""
    zest_sky_rfi_ms!(ms, sky, rfi=GPUSource[]; column="CORRECTED_DATA", gpu=true, kwargs...)

Read `column` of `ms`, replace its UVW with geometric UVW, run
`zest_sky_rfi!` (same keywords) and write the result back to `column`.
"""
function zest_sky_rfi_ms!(ms::String, sky::AbstractVector, rfi::AbstractVector=GPUSource[];
                          column::String="CORRECTED_DATA", gpu::Bool=true, kwargs...)
    vis, _, meta, bl, nrows = read_ms_fast(ms; gpu=gpu, column=column)
    mjd = ms_mjd(ms)
    geometric_uvw!(meta, mjd)
    result = zest_sky_rfi!(vis, meta, mjd, sky, rfi; kwargs...)
    write_ms_fast!(ms, vis, bl, nrows; column=column)
    result
end
