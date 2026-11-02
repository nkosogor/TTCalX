#==============================================================================#
#      Kept sources: modelled while peeling, but not subtracted                #
#==============================================================================#
#
# Per-channel direction-dependent gains absorb ("sponge") any bright source that
# is not in the model: Jupiter's decametric bursts (up to ~10^5 Jy), Per A, 3C134,
# ... A source as bright as a direction's apparent model can be followed with a
# few per cent of gain change, which smoothness or phase priors cannot tell from
# the direction's own beam and ionospheric errors (the core is compact, so another
# source's fringe barely rotates across a subband).  What separates the sources
# is their position, so the remedy is to put them in the model.
#
# A kept source is a point at a fixed position whose apparent coherency (4
# complex numbers per channel; no antenna gains, so it cannot sponge) is fitted
# by least squares on its unit fringe.  During the peel the kept sources are
# refitted after every pass and taken out of the data while the directions are
# solved, so no direction's gains need them; at the end they are put back
# (`:cosolve`).  `:both` also restores (as `restore_point_sources!`) the part of
# the kept fringes the directions still subtracted.  Jupiter's position is found
# on the data's Stokes V (names starting with "Jupiter").

"""Kept-source state of one dataset (see `kept_sources`)."""
mutable struct KeptSources
    names::Vector{String}
    fringes::Vector{Any}                  # unit fringes (Nbase × Nfreq), on the data's device
    w::Any                                # samples used in the fits
    K::NTuple{4,Any}                      # current kept component (xx, xy, yx, yy)
    C::Vector{NTuple{4,Vector{ComplexF64}}}
    offsets::Vector{Tuple{Float64,Float64}}
end

"""
    kept_sources(vis, meta, mjd, sources; minuvw=5.0, vsearch_deg=0.5) -> KeptSources

Unit fringes of the `sources` above the horizon (exact model positions; names starting
with "Jupiter" are moved to the Stokes V peak of `vis` within `vsearch_deg`), with an
empty kept component.  Call before peeling (the V search uses the data as given).
"""
function kept_sources(vis::GPUVisibilities, meta::GPUMetadata, mjd::Float64, sources::AbstractVector;
                      minuvw::Real=5.0, vsearch_deg::Real=0.5)
    gpu = _is_gpu(vis.xx)
    w = _uv_weights(meta, vis.flags, minuvw)
    names = String[]; fr = Any[]; offs = Tuple{Float64,Float64}[]
    vdiff = nothing
    for s in sources
        source_elevation_deg(s, meta) > 0 || continue
        coh = GPUVisibilities(meta.Nbase, meta.Nfreq; gpu=gpu)
        genvis_exact!(coh, meta, s, mjd)
        K = _unit_fringe(coh.xx)
        d = (0.0, 0.0)
        if startswith(get_name(s), "Jupiter") && vsearch_deg > 0
            vdiff === nothing && (vdiff = vis.xy .- vis.yx)
            K, d = _shift_to_peak(K, vdiff, w, meta, vsearch_deg)
        end
        push!(names, get_name(s)); push!(fr, K); push!(offs, d)
    end
    z() = (a = similar(vis.xx); fill!(a, zero(eltype(a))); a)
    KeptSources(names, fr, w, (z(), z(), z(), z()), NTuple{4,Vector{ComplexF64}}[], offs)
end

"""Per-channel Σ_w conj(a) b (host vector)."""
_kept_inner(a, b, w) = vec(Array(sum(ifelse.(w, conj.(a) .* b, zero(ComplexF64)); dims=1)))

"""
    fit_kept(ks, R; others=[]) -> per-source (xx, xy, yx, yy) per-channel coherencies

Least squares of each correlation of `R` (a 4-tuple of Nbase × Nfreq arrays) on the kept
fringes plus `others` (e.g. the peeled directions' xx models, so that their flux does not
leak into the kept sources), per channel, with unit-scaled columns and a tiny ridge.
"""
function fit_kept(ks::KeptSources, R; others=[])
    basis = vcat(ks.fringes, collect(others))
    nk, nb, Nf = length(ks.fringes), length(basis), size(R[1], 2)
    G = zeros(ComplexF64, nb, nb, Nf)
    for k in 1:nb, l in k:nb
        g = _kept_inner(basis[k], basis[l], ks.w)
        G[k, l, :] .= g; G[l, k, :] .= conj.(g)
    end
    C = [ntuple(_ -> zeros(ComplexF64, Nf), 4) for _ in 1:nk]
    for q in 1:4
        rhs = reduce(hcat, [_kept_inner(b, R[q], ks.w) for b in basis])       # Nf × nb
        for c in 1:Nf
            d = sqrt.(max.(real.([G[k, k, c] for k in 1:nb]), 0.0))
            any(>(0), d) || continue
            d = max.(d, 1e-12 * maximum(d))
            x = ((G[:, :, c] ./ (d * d') + 1e-10I) \ (rhs[c, :] ./ d)) ./ d
            for k in 1:nk; C[k][q][c] = x[k]; end
        end
    end
    C
end

"""Set the kept component to Σ_k C_k fringe_k."""
function set_kept!(ks::KeptSources, C)
    gpu = _is_gpu(ks.K[1])
    for q in 1:4
        fill!(ks.K[q], zero(ComplexF64))
        for k in eachindex(ks.fringes)
            c = reshape(C[k][q], 1, :)
            ks.K[q] .+= (gpu ? CuArray(c) : c) .* ks.fringes[k]
        end
    end
    ks.C = C
    ks
end

_visq(v::GPUVisibilities) = (v.xx, v.xy, v.yx, v.yy)

"""
    update_kept!(ks, vis; others=[])

`vis` holds the residual with the kept component taken out: put it back, refit it and
take the new one out.
"""
function update_kept!(ks::KeptSources, vis::GPUVisibilities; others=[])
    for (q, a) in enumerate(_visq(vis)); a .+= ks.K[q]; end
    set_kept!(ks, fit_kept(ks, _visq(vis); others=others))
    for (q, a) in enumerate(_visq(vis)); a .-= ks.K[q]; end
    ks
end

"""Band-mean Stokes I and V (Jy) of each kept source's fitted coherency."""
kept_fluxes(ks::KeptSources) = [ks.names[k] => (real(sum(C[1]) + sum(C[4])) / 2 / length(C[1]),
                                                real(-1im * (sum(C[2]) - sum(C[3]))) / 2 / length(C[1]))
                                for (k, C) in enumerate(ks.C)]
