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

#==============================================================================#
#        Restore: give back what peeling took from a known point source        #
#==============================================================================#

"""Mask of the cross-correlation samples on baselines ≥ `minuvw` λ that are not flagged."""
function _uv_weights(meta::GPUMetadata, flags, minuvw::Real)
    bl = Array(meta.baselines); pos = Array(meta.antenna_positions)
    L = [norm(pos[:, bl[2, b]] .- pos[:, bl[1, b]]) for b in axes(bl, 2)]
    lam = SPEED_OF_LIGHT ./ Array(meta.channels)[1:size(flags, 2)]
    w = (L .>= minuvw .* lam') .& (bl[1, :] .!= bl[2, :])
    (_is_gpu(flags) ? CuArray(w) : w) .& .!flags
end

"""Unit-modulus fringe K = M/|M| of a point-source model (0 where M = 0)."""
_unit_fringe(m) = ifelse.(abs.(m) .> 0, m ./ max.(abs.(m), floatmin(Float64)), zero(ComplexF64))

"""
    _shift_to_peak(K, S, w, meta, max_offset_deg) -> (K', (δl, δm))

Fringe K moved by the offset (δl, δm) within `max_offset_deg` that maximises
|Σ K̄ e^{-iφ} S| over the samples `w` (φ = 2π(u δl + v δm)ν/c): a coarse grid on
every 4th channel, then a 5× finer grid on all.  K' = K e^{iφ}.
"""
function _shift_to_peak(K, S, w, meta, max_offset_deg)
    nu = reshape(meta.channels[1:size(K, 2)], 1, :)
    u = reshape(meta.uvw[1, :], :, 1) .* (2π / SPEED_OF_LIGHT) .* nu
    v = reshape(meta.uvw[2, :], :, 1) .* (2π / SPEED_OF_LIGHT) .* nu
    P = ifelse.(w, conj.(K) .* S, zero(ComplexF64))
    sub = 1:4:size(K, 2)
    coarse = (P[:, sub], u[:, sub], v[:, sub])
    score(P, u, v, d) = abs(sum(Broadcast.instantiate(
        Broadcast.broadcasted((p, a, b) -> p * cis(-(a * d[1] + b * d[2])), P, u, v))))
    r = deg2rad(max_offset_deg)
    best = (0.0, 0.0)
    for (step, n, args) in ((r / 10, 10, coarse), (r / 50, 5, (P, u, v)))
        c = best; bs = -1.0
        for i in -n:n, j in -n:n
            d = (c[1] + i * step, c[2] + j * step)
            hypot(d...) <= r * (1 + 1e-9) || continue
            sc = score(args..., d)
            sc > bs && (bs = sc; best = d)
        end
    end
    K .* cis.(u .* best[1] .+ v .* best[2]), best
end

"""
    restore_point_sources!(vis, before, meta, mjd, sources; others=[], minuvw=5.0,
                           offset_deg=0.0) -> Vector of (C, offset)

Add back what peeling took from the point `sources` (e.g. Jupiter, given at its
ephemeris position).  The subtracted signal `before - vis` is fitted per
channel and correlation by least squares as Σ_k c_k B_k over the unit fringes
of the `sources` and the models `others` of the peeled directions (xx
visibilities), so the peeled sources' own flux does not leak into a source's
estimate.  The sources' fits c K are added to `vis`; C = the per-channel
coherencies (xx, xy, yx, yy) in Jy.

`offset_deg > 0` first moves each source to the position within that radius
where the original data's Stokes V (xy - yx) peaks: for a strongly circularly
polarised source such as Jupiter's decametric emission, whose position in the
data can differ from the model's by 10-25'.  Leave it 0 for other sources.
"""
function restore_point_sources!(vis::GPUVisibilities, before::GPUVisibilities, meta::GPUMetadata,
                                mjd::Float64, sources::AbstractVector; others=[],
                                minuvw::Real=5.0, offset_deg::Real=0.0)
    gpu = _is_gpu(vis.xx)
    w = _uv_weights(meta, vis.flags, minuvw)
    offsets = Tuple{Float64,Float64}[]
    fr = map(sources) do s
        coh = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=gpu)
        genvis_exact!(coh, meta, s, mjd)
        K = _unit_fringe(coh.xx)
        if offset_deg > 0
            K, d = _shift_to_peak(K, before.xy .- before.yx, w, meta, offset_deg)
            push!(offsets, d)
        else
            push!(offsets, (0.0, 0.0))
        end
        K
    end
    basis = vcat(fr, collect(others))
    nsrc, nk, Nf = length(fr), length(basis), size(vis.xx, 2)
    masked(x) = vec(Array(sum(ifelse.(w, x, zero(ComplexF64)); dims=1)))
    G = zeros(ComplexF64, nk, nk, Nf)                    # normal equations, built on the device
    for k in 1:nk, l in k:nk
        G[k, l, :] .= masked(conj.(basis[k]) .* basis[l]); G[l, k, :] .= conj.(G[k, l, :])
    end
    C = [ntuple(_ -> zeros(ComplexF64, Nf), 4) for _ in 1:nsrc]
    for (q, f) in enumerate((:xx, :xy, :yx, :yy))
        sub = getfield(before, f) .- getfield(vis, f)
        rhs = reduce(hcat, [masked(conj.(b) .* sub) for b in basis])          # Nf × nk
        for c in 1:Nf
            d = sqrt.(max.(real.([G[k, k, c] for k in 1:nk]), 0.0))           # column norms
            any(>(0), d) || continue                                           # no data
            d = max.(d, 1e-12 * maximum(d))
            Gs = G[:, :, c] ./ (d * d')                                        # unit columns
            x = ((Gs + 1e-10I) \ (rhs[c, :] ./ d)) ./ d                        # tiny relative ridge
            for k in 1:nsrc; C[k][q][c] = x[k]; end
        end
        for k in 1:nsrc
            getfield(vis, f) .+= reshape(gpu ? CuArray(C[k][q]) : C[k][q], 1, :) .* fr[k]
        end
    end
    collect(zip(C, offsets))
end

"""Whether `g` (a previous result's `gains`) may start this solve: same channels, recent."""
function _warm_usable(g, meta, mjd, max_age_s)
    g === nothing && return false
    if length(g.channels) != meta.Nfreq || !isapprox(g.channels, Array(meta.channels); rtol=1e-9)
        log_detail("warm start skipped: previous gains are for other channels")
        return false
    end
    age = abs(mjd - g.mjd) * 86400
    age <= max_age_s || (log_detail(@sprintf("warm start skipped: previous gains are %.0f s old", age)); return false)
    true
end

"""Copy gains `g0` into `cal` when they have the same shape and Jones type."""
function _warm_start!(cal, g0, name)
    g0 === nothing && return cal
    if size(g0.xx) != size(cal.xx) || g0.is_diagonal != cal.is_diagonal
        log_detail("warm start skipped for $name: gains of another shape")
        return cal
    end
    foreach(f -> copyto!(getfield(cal, f), getfield(g0, f)), (:xx, :xy, :yx, :yy))
    cal
end

"""Catalog Stokes I (Jy) of a sky source at `nu` (sum over components)."""
_intrinsic_flux(s::GPUMultiSource, nu) = sum(_intrinsic_flux(c, nu) for c in s.components)
_intrinsic_flux(s, nu) = hasproperty(s, :spectrum) ? Float64(s.spectrum(nu)[1]) : 0.0

"""True when the whole band lies above `mhz`."""
_band_above(meta, mhz) = minimum(Array(meta.channels)) > mhz * 1e6

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
- `smooth_flux = 0.0` (Jy, off): tie each sky direction's per-channel gains to one band
  gain (`solve_gains!`): directions fainter than this cannot absorb other sources.
  `rfi_smooth_flux = smooth_flux` the same for the RFI emitters; sky sources with catalog
  flux ≥ `smooth_exempt_flux = Inf` (e.g. 1e4 for Cas A, Cyg A) always keep per-channel gains
- `restore = GPUSource[]`: point sources (e.g. Jupiter at its ephemeris position) given back
  after peeling (`restore_point_sources!`), only when the band starts below
  `restore_below_mhz = Inf`; `restore_offset_deg = 0.0` position search radius on Stokes V

- `precision = :double` or `:mixed` (normal equations in single precision, see `solve_gains!`)
- `initial_gains = nothing` or a previous result's `gains`: starting gains per source name, used
  only when they were solved on the same channels within `warm_max_age_s = 60` seconds and have
  the same shape (otherwise identity, with a log line saying why)
- `cache_model_sq = (precision == :mixed)`: build each direction's square model once per pass
  instead of per solve (faster; ~190 MB per direction in :mixed, 380 MB in :double)

Returns `(peeled, skipped, flagged_zeros, restored, gains)`: names of the subtracted sources,
names of the sky sources below `min_elevation`, newly flagged zero samples,
`name => (C, offset)` for each restored source, and `gains = (channels, mjd, cals)` with
`cals = Dict(name => final gains)` (the live calibration objects; a source the guard rejected
keeps its starting gains).
"""
function zest_sky_rfi!(vis::GPUVisibilities, meta::GPUMetadata, mjd::Float64,
                       sky::AbstractVector, rfi::AbstractVector=GPUSource[];
                       rfi_mode::Symbol=:after, solver::Symbol=:rk4, maxiter::Integer=5,
                       tolerance::Real=1e-4, minuvw::Real=5.0, peeliter::Integer=3,
                       max_gain::Real=100.0, min_elevation::Real=0.0, diagonal::Bool=false,
                       reg=nothing, flag_zeros::Bool=true, report=nothing,
                       smooth_flux::Real=0.0, rfi_smooth_flux::Real=smooth_flux,
                       smooth_exempt_flux::Real=Inf, restore::AbstractVector=GPUSource[],
                       restore_below_mhz::Real=Inf, restore_offset_deg::Real=0.0,
                       precision::Symbol=:double, initial_gains=nothing, warm_max_age_s::Real=60.0,
                       cache_model_sq::Bool=(precision == :mixed))
    rfi_mode in (:after, :joint) || throw(ArgumentError("rfi_mode must be :after or :joint"))
    precision in (:double, :mixed) || throw(ArgumentError("precision must be :double or :mixed"))
    smooth_exempt_flux > 0 || throw(ArgumentError("smooth_exempt_flux must be > 0 (Inf: none exempt)"))
    min(smooth_flux, rfi_smooth_flux) >= 0 || throw(ArgumentError("smooth fluxes must be ≥ 0"))
    (smooth_flux > 0 || rfi_smooth_flux > 0) && reg !== nothing &&
        throw(ArgumentError("smooth_flux cannot be combined with reg"))
    restore_offset_deg >= 0 || throw(ArgumentError("restore_offset_deg must be ≥ 0"))
    gpu = _is_gpu(vis.xx)
    nzero = flag_zeros ? flag_zero_visibilities!(vis) : 0
    prior = reg === nothing ? nothing :
        phase_regularization(meta, reg[1], reg[2]; lambda=length(reg) > 2 ? reg[3] : 1.0, gpu=gpu)
    nu0 = _mean(Array(meta.channels))
    smooth_of(s) = is_rfi(s) ? rfi_smooth_flux :
        _intrinsic_flux(s, nu0) >= smooth_exempt_flux ? 0.0 : smooth_flux

    up = [source_elevation_deg(s, meta) >= min_elevation for s in sky]
    restored = _band_above(meta, restore_below_mhz) ? GPUSource[] :
        [s for s in restore if source_elevation_deg(s, meta) >= min_elevation]
    before = isempty(restored) ? nothing : deepcopy_gpu(vis)
    peeled_models = []                     # xx models of the subtracted directions (restore basis)
    warm = _warm_usable(initial_gains, meta, mjd, warm_max_age_s)
    solved = Pair{String,GPUCalibration}[]  # final gains of every solved direction
    passes = rfi_mode == :joint ? [vcat(sky[up], rfi)] : [sky[up], rfi]
    wrap = diagonal ? GPUPeelingSource : GPUZestingSource
    peeled = String[]
    for sources in passes
        isempty(sources) && continue
        cals = [calibration_type(wrap(s), meta.Nant, meta.Nfreq; gpu=gpu) for s in sources]
        for (s, cal) in zip(sources, cals)          # warm start, e.g. from the previous snapshot
            warm && _warm_start!(cal, get(initial_gains.cals, get_name(s), nothing), get_name(s))
        end
        append!(solved, get_name.(sources) .=> cals)
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
        sflux = smooth_of.(sources)
        # the square models are constant over the peel passes: built once per direction
        # (Nant² Nfreq × 4 per direction: 190 MB at 352 antennas, 48 channels, :mixed)
        sqcache = IdDict{Any,Any}()
        meas_sq = GPUSquareVisibilities(meta.Nant, meta.Nfreq; gpu=gpu,
                                        T=precision == :mixed ? ComplexF32 : ComplexF64)
        solve = (cal, v, coh) -> begin
            msq = cache_model_sq ?
                get!(() -> square_model(coh, meta; minuvw=minuvw, precision=precision), sqcache, coh) :
                nothing
            converged, niters, _ = solve_gains!(cal, v, coh, meta; solver=solver, maxiter=maxiter,
                                                tolerance=tolerance, minuvw=minuvw, reg=prior,
                                                smooth_flux=sflux[findfirst(c -> c === cal, cals)],
                                                precision=precision, model_sq=msq, meas_sq=meas_sq)
            converged, niters
        end
        sub = guarded_zest!(vis, meta, cohs, cals, solve; peeliter=peeliter,
                            max_gain=max_gain, report=rep)
        append!(peeled, get_name.(sources[sub]))
        before === nothing || append!(peeled_models, [c.xx for c in cohs[sub]])
    end
    res = isempty(restored) ? [] :
        restore_point_sources!(vis, before, meta, mjd, restored; others=peeled_models,
                               minuvw=minuvw, offset_deg=restore_offset_deg)
    for (s, (C, d)) in zip(restored, res)
        log_detail(@sprintf("%s: restored %.1f Jy (band-mean Stokes I), offset %.1f', %.1f'",
                            get_name(s), real(_mean(C[1]) + _mean(C[4])),
                            rad2deg(d[1]) * 60, rad2deg(d[2]) * 60))
    end
    (peeled=peeled, skipped=get_name.(sky[.!up]), flagged_zeros=nzero,
     restored=[get_name(s) => r for (s, r) in zip(restored, res)],
     gains=(channels=Array(meta.channels), mjd=mjd, cals=Dict(solved)))
end

"""
    zest_sky_rfi_ms!(ms, sky, rfi=GPUSource[]; column="CORRECTED_DATA", gpu=true, kwargs...)

Read `column` of `ms`, replace its UVW with geometric UVW, run
`zest_sky_rfi!` (same keywords) and write the result back to `column`.
"""
function zest_sky_rfi_ms!(ms::String, sky::AbstractVector, rfi::AbstractVector=GPUSource[];
                          column::String="CORRECTED_DATA", gpu::Bool=true, kwargs...)
    t_read = @elapsed begin
        vis, _, meta, bl, nrows = read_ms_fast(ms; gpu=gpu, column=column)
        mjd = ms_mjd(ms)
        geometric_uvw!(meta, mjd)
    end
    t_zest = @elapsed (result = zest_sky_rfi!(vis, meta, mjd, sky, rfi; kwargs...))
    t_write = @elapsed write_ms_fast!(ms, vis, bl, nrows; column=column)
    log_detail(@sprintf("timing: read %.2f s, zest %.2f s, write %.2f s", t_read, t_zest, t_write))
    result
end
