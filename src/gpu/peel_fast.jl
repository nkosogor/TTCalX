#==============================================================================#
#            Fast peel: zest_sky_rfi!(...; peel_fast=true)                     #
#==============================================================================#
#
# The same peel as the default zest_sky_rfi! (rk4, :mixed, fast_kernels!, the residual guard), with
# the work rearranged on the GPU.  Off (the default) none of this code runs and the peel is the
# default one bit for bit.  On, the result is another rounding of the default peel, in about 1/3 of
# the time: on the OVRO-LWA validation snapshots it agrees to Float32 rounding at 55 and 82 MHz
# (~1e-8 - 1e-6 relative) and to ~1e-4 rms / ~1e-3 max relative at 23 MHz, where the ill-conditioned
# solves amplify rounding and a guard decision (RFI) occasionally flips, as with fast_kernels! on vs off;
# deep images at the production floor.  The work rearranged:
#
#   baseline compaction   only baselines with an unflagged sample are peeled (padded to a multiple of
#                         1024 rows); the accepted directions are then also subtracted, with their
#                         final gains, from the fully flagged rows, so every row is peeled
#   antenna compaction    antennas without unflagged data are left out of the square matrices and
#                         the normal equations (their step is exactly 0, as in the default)
#   square matrices       32x32 tiles through shared memory, only the tiles I <= J the tiled normal
#                         equations read, the short-baseline mask built once per call; no zero
#                         fills of buffers that are fully overwritten
#   precision             ComplexF32 gain copies in the normal equations, the 2x2 step solve in
#                         Float32, model x gain products of the add / subtract in Float32, model
#                         fringes with the phase reduced mod 2pi in Float64 and sincos in Float32;
#                         ComplexF32 visibilities when the caller passes them (the validated setup)
#   RK4                   ~9 launches per iteration: the step kernel also writes the RK stage and
#                         its Float32 copy, one kernel per channel does the update, the stop test
#                         and the mask; the early stop is read from host-mapped memory one
#                         iteration late (the extra iteration is a no-op: every channel frozen)
#   guard                 the gain check (finite, max|g|), the conditional subtract and the residual
#                         powers on the device, the next direction's add-back fused with this one's
#                         subtraction, one host wait per direction update
#
# Process-wide state (one fast peel per process at a time, one GPU per process):
#   PEEL_FAST            true only while a fast zest_sky_rfi! runs; a second fast call meanwhile, or
#                        a default zest_sky_rfi! call, throws
#   host-mapped memory   512 KiB of pinned host memory mapped into the device, allocated once on the
#                        device current at the first fast peel (a fast peel on another device throws)
#   CUDA events          a few reusable events of that context
#   Float32 gain copies  four Na x Nf ComplexF32 arrays per role (_g32!), kept for the process
#   per-call caches      the baseline pair table and the short-baseline mask: built in the call and
#                        dropped at its end (nothing is keyed on array identity across calls)

const PEEL_FAST = Ref(false)
const _PEEL_FAST_BUSY = Threads.Atomic{Bool}(false)

"""Largest `maxiter` of a fast peel (64 host-mapped slots per wait cycle, events 1:maxiter + 1)."""
const FAST_MAXITER = 63

const _BLC_PAD = 1024                  # compacted baseline count rounded up to a multiple of this

# per-call caches (reset at the start and end of every fast call)
const _SQ_TABLE = Ref{Any}(nothing)    # (baselines, live, device pair table)
const _SQ_MASK = Ref{Any}(nothing)     # (baselines, positions, channels, minuvw, device mask)
# the RFI models' phase centre ŝ₀ (uvw_geometry) fitted once on the full metadata of the call: the compacted
# rows (and the few fully flagged rows) would fit it from a subset (underdetermined for 1-2 dead rows)
const _RFI_S0 = Ref{Any}(nothing)
_fast_reset_caches!() = (_SQ_TABLE[] = nothing; _SQ_MASK[] = nothing; _RFI_S0[] = nothing; nothing)

const _G32 = Dict{Any,Any}()
"""ComplexF32 copies (role `s`) of the four gain arrays `g` (buffers kept per role, shape and device)."""
function _g32!(g, s::Int=1)
    key = (s, size(g[1]), CUDA.device())
    b = get!(() -> ntuple(_ -> CuArray{ComplexF32}(undef, size(g[1])), 4), _G32, key)
    for q in 1:4; b[q] .= ComplexF32.(g[q]); end
    b
end

#------------------------------------------------------------------------------#
# Antenna compaction                                                           #
#------------------------------------------------------------------------------#

"""Live antennas of a compacted solve: `live` (device) / `live_host` (compact → antenna), `comp` (antenna → compact or 0)."""
struct AntennaCompaction{V<:AbstractVector{Int32}}
    live::V
    live_host::Vector{Int32}
    comp::V
    N::Int
end

"""
    compact_antennas(vis, meta) -> AntennaCompaction or nothing

The live antennas of `vis` (an unflagged cross-correlation sample in some channel); `nothing` when
every antenna is live (nothing to compact) or none is.  GPU visibilities.
"""
function compact_antennas(vis::GPUVisibilities, meta::GPUMetadata)
    live = _live_antennas(vis, meta)
    (all(live) || !any(live)) && return nothing
    lh = Int32.(findall(live))
    comp = zeros(Int32, meta.Nant)
    comp[lh] .= 1:length(lh)
    AntennaCompaction(CuArray(lh), lh, CuArray(comp), length(lh))
end

"""Live antennas (host BitVector): an unflagged cross-correlation sample of `vis` in some channel."""
function _live_antennas(vis::GPUVisibilities, meta::GPUMetadata)
    alive_bl = Array(vec(any(.!vis.flags; dims=2)))
    bl = Array(meta.baselines)
    live = falses(meta.Nant)
    for α in axes(bl, 2)
        if alive_bl[α] && bl[1, α] != bl[2, α]
            live[bl[1, α]] = true; live[bl[2, α]] = true
        end
    end
    live
end

#------------------------------------------------------------------------------#
# Square matrices by tiles                                                     #
#------------------------------------------------------------------------------#
# A block takes the tile pair (I, J), I ≤ J, of one channel: it reads the baselines of entries [r, c]
# (r in tile I, c in tile J) with neighbouring threads on neighbouring c, and writes the upper tile
# [r, c] through shared memory, coalesced (the tiled normal equations read only the tiles I ≤ J).
# `table[c, r]` is the baseline of entry [r, c]: +α when baseline α is (r, c), -α when it is (c, r)
# (the conjugate-transposed entry), 0 when no baseline has the pair (written zero).  The values are
# makesquare_masked_kernel!'s (one baseline per antenna pair, as it assumes).

"""Host table of the tile kernel for baselines `bl` (2×Nbase) of `Na` antennas, `live` ones (all when `nothing`)."""
function _pair_table_host(bl::AbstractMatrix, Na::Int, live=nothing)
    comp = live === nothing ? collect(Int32, 1:Na) : zeros(Int32, Na)
    live === nothing || (comp[live] .= 1:length(live))
    N = live === nothing ? Na : length(live)
    tab = zeros(Int32, N, N)
    for α in axes(bl, 2)
        r = comp[bl[1, α]]; c = comp[bl[2, α]]
        (r == 0 || c == 0) && continue
        tab[c, r] = α                     # entry [r, c] = baseline α
        r != c && (tab[r, c] = -α)        # entry [c, r] = its conjugate transpose
    end
    tab
end

"""Device pair table of `meta` (antennas `live`, all when `nothing`); cached within a fast call."""
function _pair_table(meta::GPUMetadata, live=nothing)
    c = _SQ_TABLE[]
    c !== nothing && c[1] === meta.baselines && isequal(c[2], live) && return c[3]
    t = CuArray(_pair_table_host(Array(meta.baselines), meta.Nant, live))
    _SQ_TABLE[] = (meta.baselines, live === nothing ? nothing : copy(live), t)
    t
end

const _SQ_TY = 8                    # threads (32, 8): each thread takes 4 rows of the tile

"""0-based tile pair (I, J), I ≤ J, of block `b` (1-based) among the ntile(ntile+1)/2 pairs."""
@inline function _tile_pair(b, ntile)
    I = 0; rem = b - 1
    while rem >= ntile - I
        rem -= ntile - I; I += 1
    end
    I, I + rem
end

"""Store one correlation `v` (the thread's 4 rows) of the upper tile [r, c] through shared memory."""
@inline function _sq_upper!(out, sh, v, I, J, tx, ty, N, β)
    for k in 1:4
        @inbounds sh[tx, ty + _SQ_TY * (k - 1)] = v[k]          # sh[c, r]
    end
    sync_threads()
    r = I * _NE_TILE + tx                                       # threads along r: coalesced
    for k in 1:4
        cl = ty + _SQ_TY * (k - 1)
        cc = J * _NE_TILE + cl
        if r <= N && cc <= N
            @inbounds out[r, cc, β] = sh[cl, tx]
        end
    end
    sync_threads()
    nothing
end

# keep mask of every (baseline, channel): not an autocorrelation and not shorter than minuvw wavelengths
# (makesquare_masked_kernel!'s Float64 test)
function fast_smask_kernel!(sm, baselines, pos, channels, minuvw, Nb, Nf)
    i = thread_index_1d()
    if i <= Nb * Nf
        α = ((i - 1) % Nb) + 1; β = ((i - 1) ÷ Nb) + 1
        a1 = baselines[1, α]; a2 = baselines[2, α]
        keep = a1 != a2
        if keep && minuvw > 0
            L = sqrt((pos[1, a1] - pos[1, a2])^2 + (pos[2, a1] - pos[2, a2])^2 + (pos[3, a1] - pos[3, a2])^2)
            keep = L >= minuvw * (SPEED_OF_LIGHT / channels[β])
        end
        sm[α, β] = keep ? 0x01 : 0x00
    end
    return nothing
end

"""Short-baseline keep mask of `meta` (Nbase × Nf, UInt8); cached within a fast call."""
function _short_mask(meta, minuvw, Nf)
    c = _SQ_MASK[]
    if c !== nothing && c[1] === meta.baselines && c[2] === meta.antenna_positions && c[3] === meta.channels &&
       c[4] == minuvw && size(c[5]) == (meta.Nbase, Nf)
        return c[5]
    end
    sm = CuArray{UInt8}(undef, meta.Nbase, Nf)
    blocks, threads = kernel_config_1d(meta.Nbase * Nf)
    @cuda blocks=blocks threads=threads fast_smask_kernel!(sm, meta.baselines, meta.antenna_positions, meta.channels,
                                                           Float64(minuvw), meta.Nbase, Nf)
    _SQ_MASK[] = (meta.baselines, meta.antenna_positions, meta.channels, minuvw, sm)
    sm
end

@inline function _sq_entry_m(s, vxx, vxy, vyx, vyy, flags, sm, β, ::Type{T}) where {T}
    z = zero(T)
    s == 0 && return (z, z, z, z)
    α = abs(s)
    keep = !flags[α, β] && sm[α, β] != 0x00
    xx = keep ? T(vxx[α, β]) : z; xy = keep ? T(vxy[α, β]) : z
    yx = keep ? T(vyx[α, β]) : z; yy = keep ? T(vyy[α, β]) : z
    s > 0 ? (xx, xy, yx, yy) : (conj(xx), conj(yx), conj(xy), conj(yy))
end

@inline function _sq_row_m(k, I, J, tx, ty, N, lower, sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, β,
                           ::Type{T}) where {T}
    z = zero(T)
    r = I * _NE_TILE + ty + _SQ_TY * (k - 1)
    c = J * _NE_TILE + tx
    (r <= N && c <= N) || return (z, z, z, z)
    s = table[c, r]
    e = _sq_entry_m(s, vxx, vxy, vyx, vyy, flags, sm, β, T)
    if lower                            # lower tile: [c, r] = [r, c]ᴴ (zero without a baseline)
        l = s == 0 ? e : (conj(e[1]), conj(e[3]), conj(e[2]), conj(e[4]))
        sxx[c, r, β] = l[1]; sxy[c, r, β] = l[2]; syx[c, r, β] = l[3]; syy[c, r, β] = l[4]
    end
    e
end

function fast_sqtile_kernel!(sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, N, ntile, upper::Bool)
    T = eltype(sxx)
    β = blockIdx().y
    I, J = _tile_pair(blockIdx().x, ntile)
    tx = threadIdx().x; ty = threadIdx().y
    lower = I != J && !upper
    e1 = _sq_row_m(1, I, J, tx, ty, N, lower, sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, β, T)
    e2 = _sq_row_m(2, I, J, tx, ty, N, lower, sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, β, T)
    e3 = _sq_row_m(3, I, J, tx, ty, N, lower, sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, β, T)
    e4 = _sq_row_m(4, I, J, tx, ty, N, lower, sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags, sm, table, β, T)
    sh = CuDynamicSharedArray(T, (_NE_TILE + 1, _NE_TILE))     # padded: no bank conflicts on the transpose
    _sq_upper!(sxx, sh, (e1[1], e2[1], e3[1], e4[1]), I, J, tx, ty, N, β)
    _sq_upper!(sxy, sh, (e1[2], e2[2], e3[2], e4[2]), I, J, tx, ty, N, β)
    _sq_upper!(syx, sh, (e1[3], e2[3], e3[3], e4[3]), I, J, tx, ty, N, β)
    _sq_upper!(syy, sh, (e1[4], e2[4], e3[4], e4[4]), I, J, tx, ty, N, β)
    return nothing
end

"""
    _fast_square!(sq, vis, meta, minuvw, compact) -> sq

The square matrices `square_visibilities!` writes, only the tiles I ≤ J (upper), of the antennas of
`compact` (an `AntennaCompaction`, leading N×N block) or of all antennas (`nothing`).  Not synchronised.
"""
function _fast_square!(sq, vis, meta, minuvw, compact)
    live = compact === nothing ? nothing : compact.live_host
    N = compact === nothing ? meta.Nant : compact.N
    Nf = Nfreq(vis)
    table = _pair_table(meta, live)
    sm = _short_mask(meta, minuvw, Nf)
    ntile = cld(N, _NE_TILE)
    T = eltype(sq.xx)
    @cuda blocks=(ntile * (ntile + 1) ÷ 2, Nf) threads=(_NE_TILE, _SQ_TY) shmem=(_NE_TILE + 1) * _NE_TILE * sizeof(T) fast_sqtile_kernel!(
        sq.xx, sq.xy, sq.yx, sq.yy, vis.xx, vis.xy, vis.yx, vis.yy, vis.flags, sm, table, N, ntile, true)
    sq
end

#------------------------------------------------------------------------------#
# Host-mapped memory and events                                                #
#------------------------------------------------------------------------------#

mutable struct _FastMapped
    buf::Any
    h::Vector{Float64}
    d::Any
    dev::CuDevice
end
const _FAST_MAPPED = Ref{Any}(nothing)
const _FAST_SLOTS = 1 << 16
function _fast_mapped()
    m = _FAST_MAPPED[]
    if m === nothing
        buf = CUDA.alloc(CUDA.HostMemory, _FAST_SLOTS * sizeof(Float64), CUDA.MEMHOSTALLOC_DEVICEMAP)
        h = unsafe_wrap(Array, convert(Ptr{Float64}, buf), _FAST_SLOTS)
        d = unsafe_wrap(CuArray, convert(CuPtr{Float64}, buf), _FAST_SLOTS)
        m = _FastMapped(buf, h, d, CUDA.device()); _FAST_MAPPED[] = m
    end
    m.dev == CUDA.device() ||
        error("peel_fast: the host-mapped memory belongs to $(m.dev), this peel runs on $(CUDA.device()) " *
              "(one GPU per process)")
    m
end
const _FAST_EV = CuEvent[]
function _fast_event(i)
    while length(_FAST_EV) < i; push!(_FAST_EV, CuEvent(CUDA.EVENT_DISABLE_TIMING)); end
    _FAST_EV[i]
end
_fast_wait(ev) = (while !CUDA.isdone(ev); end; nothing)      # spin: the waits are short
const _FAST_SLOT = Ref(0)            # ring of host-mapped regions of 64 slots
function _fast_region()
    _FAST_SLOT[] = (_FAST_SLOT[] + 64) % (_FAST_SLOTS - 64)
    _FAST_SLOT[]
end

#------------------------------------------------------------------------------#
# RK4 solve                                                                    #
#------------------------------------------------------------------------------#

@inline _tile_gain(gxx, gxy, gyx, gyy, a, β, ::Type{T}) where {T} = (T(gxx[a, β]), T(gxy[a, β]), T(gyx[a, β]), T(gyy[a, β]))

"""full_ne_tile_kernel! with the gains of compact antenna r at `live[r]` (all antennas: `live = nothing`)."""
function fast_ne_tile_kernel!(part, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                              active, Nant, ntile, live)
    T = eltype(Vxx)
    β = blockIdx().y
    active !== nothing && !active[β] && return nothing
    I = 0; rem = blockIdx().x - 1                       # tile pair (I, J), I ≤ J, 0-based
    while rem >= ntile - I
        rem -= ntile - I; I += 1
    end
    J = I + rem
    tx = threadIdx().x; ty = threadIdx().y; TY = blockDim().y
    r = I * _NE_TILE + tx
    zt = zero(T)
    z8 = (zt, zt, zt, zt, zt, zt, zt, zt)
    gr = r <= Nant ? _tile_gain(gxx, gxy, gyx, gyy, live === nothing ? r : Int(live[r]), β, T) : (zt, zt, zt, zt)
    racc = z8
    for k in 0:(_NE_TILE ÷ TY - 1)
        c = J * _NE_TILE + ty + TY * k
        ccon = z8
        if r <= Nant && c <= Nant
            m = (Mxx[r, c, β], Mxy[r, c, β], Myx[r, c, β], Myy[r, c, β])
            v = (Vxx[r, c, β], Vxy[r, c, β], Vyx[r, c, β], Vyy[r, c, β])
            ccon = _ne_terms(gr, m, v)                  # antenna c, i = r
            if I != J                                   # antenna r, i = c
                gc = _tile_gain(gxx, gxy, gyx, gyy, live === nothing ? c : Int(live[c]), β, T)
                rc = _ne_terms(gc, _hblock(m), _hblock(v))
                racc = map(+, racc, rc)
            end
        end
        ccon = map(_warp_sum, ccon)
        if tx == 1 && c <= Nant
            for q in 1:8
                @inbounds part[c, q, I + 1, β] = ccon[q]
            end
        end
    end
    if I != J
        sh = CuDynamicSharedArray(T, (8, _NE_TILE, TY))
        for q in 1:8
            @inbounds sh[q, tx, ty] = racc[q]
        end
        sync_threads()
        if ty == 1 && r <= Nant
            for q in 1:8
                acc = sh[q, tx, 1]
                for y in 2:TY
                    acc += sh[q, tx, y]
                end
                @inbounds part[r, q, J + 1, β] = acc
            end
        end
    end
    return nothing
end

"""Normal equations of antenna `j` (compact number), channel β, from the tile partial sums in their own precision."""
@inline function _sum_part32(part, j, β, ntile)
    ntuple(Val(8)) do q
        acc = part[j, q, 1, β]
        for t in 2:ntile
            acc += part[j, q, t, β]
        end
        acc
    end
end

"""`_full_step_ne` (no prior) with the 2x2 solve in the normal equations' own (single) precision."""
@inline function _full_step_ne32(j, β, ne, gxx, gxy, gyx, gyy)
    nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy = ne
    z = ComplexF64(0)
    det_den = dxx*dyy - dxy*dyx
    abs(det_den) > eps(Float64) || return z, z, z, z
    inv_det = one(det_den) / det_den
    inv_xx = dyy * inv_det; inv_xy = -dxy * inv_det
    inv_yx = -dyx * inv_det; inv_yy = dxx * inv_det
    sol_xx = inv_xx*nxx + inv_xy*nyx
    sol_xy = inv_xx*nxy + inv_xy*nyy
    sol_yx = inv_yx*nxx + inv_yy*nyx
    sol_yy = inv_yx*nxy + inv_yy*nyy
    return ComplexF64(conj(sol_xx)) - gxx[j, β], ComplexF64(conj(sol_yx)) - gxy[j, β],
           ComplexF64(conj(sol_xy)) - gyx[j, β], ComplexF64(conj(sol_yy)) - gyy[j, β]
end

function fast_tog32_kernel!(o1, o2, o3, o4, g1, g2, g3, g4, n)
    i = thread_index_1d()
    if i <= n
        @inbounds o1[i] = ComplexF32(g1[i]); @inbounds o2[i] = ComplexF32(g2[i])
        @inbounds o3[i] = ComplexF32(g3[i]); @inbounds o4[i] = ComplexF32(g4[i])
    end
    return nothing
end

# StefCal step k of antenna j (0 for frozen channels and dead antennas), then for stage 1 / 2 the RK stage
# x = base + 0.5 k, stage 3 x = base + k (stage 4: none), and its ComplexF32 copy y (the next NE's gains)
function fast_step_kernel!(kxx, kxy, kyx, kyy, part, gxx, gxy, gyx, gyy, b1, b2, b3, b4, x1, x2, x3, x4,
                           y1, y2, y3, y4, stage::Int, active, Nant, Nfreq, ntile, comp, lowp::Bool)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        jc = comp === nothing ? j : Int(comp[j])
        if (active !== nothing && !active[β]) || jc == 0
            z = zero(eltype(kxx))
            sxx = z; sxy = z; syx = z; syy = z
        elseif lowp
            sxx, sxy, syx, syy = _full_step_ne32(j, β, _sum_part32(part, jc, β, ntile), gxx, gxy, gyx, gyy)
        else
            ne = _sum_part(part, jc, β, ntile)
            sxx, sxy, syx, syy = _full_step_ne(j, β, ne, gxx, gxy, gyx, gyy, nothing, nothing, 0.0)
        end
        kxx[j, β] = sxx; kxy[j, β] = sxy; kyx[j, β] = syx; kyy[j, β] = syy
        if stage <= 3
            # the broadcasts x[c] .= g[c] .+ 0.5 .* k[c] / x[c] .= g[c] .+ k[c] (k read back: the stored value)
            q1 = kxx[j, β]; q2 = kxy[j, β]; q3 = kyx[j, β]; q4 = kyy[j, β]
            if stage == 3
                v1 = b1[j, β] + q1; v2 = b2[j, β] + q2; v3 = b3[j, β] + q3; v4 = b4[j, β] + q4
            else
                v1 = b1[j, β] + 0.5 * q1; v2 = b2[j, β] + 0.5 * q2; v3 = b3[j, β] + 0.5 * q3; v4 = b4[j, β] + 0.5 * q4
            end
            x1[j, β] = v1; x2[j, β] = v2; x3[j, β] = v3; x4[j, β] = v4
            y1[j, β] = ComplexF32(v1); y2[j, β] = ComplexF32(v2); y3[j, β] = ComplexF32(v3); y4[j, β] = ComplexF32(v4)
        end
    end
    return nothing
end

const _FAST_FT = 256
@inline _rk4_gnew(g, a, b, c, d) = g + (a + 2 * b + 2 * c + d) / 6

# The end of an RK4 iteration, one block per channel β: gnew, the stop test ||gnew - g|| / ||g|| < tol, the
# masked update, the mask and the ComplexF32 gains; the last block writes the number of channels still
# iterating to the host-mapped slot `slot`.  fast_final1_kernel!: one thread per antenna (Na ≤ 1024).
function fast_final_kernel!(g1, g2, g3, g4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4,
                            y1, y2, y3, y4, mask, Na, tol, cnt, ci, hm, slot)
    β = blockIdx().x
    tid = threadIdx().x
    nb = gridDim().x
    m = mask[β]
    dsum = 0.0; gsum = 0.0
    j = tid
    while j <= Na
        @inbounds begin
            n1 = _rk4_gnew(g1[j, β], a1[j, β], b1[j, β], c1[j, β], d1[j, β])
            n2 = _rk4_gnew(g2[j, β], a2[j, β], b2[j, β], c2[j, β], d2[j, β])
            n3 = _rk4_gnew(g3[j, β], a3[j, β], b3[j, β], c3[j, β], d3[j, β])
            n4 = _rk4_gnew(g4[j, β], a4[j, β], b4[j, β], c4[j, β], d4[j, β])
            dsum += abs2(n1 - g1[j, β]) + abs2(n2 - g2[j, β]) + abs2(n3 - g3[j, β]) + abs2(n4 - g4[j, β])
            gsum += abs2(g1[j, β]) + abs2(g2[j, β]) + abs2(g3[j, β]) + abs2(g4[j, β])
        end
        j += _FAST_FT
    end
    sh = CuStaticSharedArray(Float64, (_FAST_FT, 2))
    @inbounds sh[tid, 1] = dsum; @inbounds sh[tid, 2] = gsum
    sync_threads()
    s = _FAST_FT ÷ 2
    while s > 0
        if tid <= s
            @inbounds sh[tid, 1] += sh[tid + s, 1]
            @inbounds sh[tid, 2] += sh[tid + s, 2]
        end
        sync_threads()
        s ÷= 2
    end
    done = sqrt(sh[1, 1]) / sqrt(sh[1, 2]) < tol
    j = tid
    while j <= Na
        @inbounds begin
            if m
                g1[j, β] = _rk4_gnew(g1[j, β], a1[j, β], b1[j, β], c1[j, β], d1[j, β])
                g2[j, β] = _rk4_gnew(g2[j, β], a2[j, β], b2[j, β], c2[j, β], d2[j, β])
                g3[j, β] = _rk4_gnew(g3[j, β], a3[j, β], b3[j, β], c3[j, β], d3[j, β])
                g4[j, β] = _rk4_gnew(g4[j, β], a4[j, β], b4[j, β], c4[j, β], d4[j, β])
            end
            y1[j, β] = ComplexF32(g1[j, β]); y2[j, β] = ComplexF32(g2[j, β])
            y3[j, β] = ComplexF32(g3[j, β]); y4[j, β] = ComplexF32(g4[j, β])
        end
        j += _FAST_FT
    end
    tid == 1 && _fast_count_active!(m & !done, mask, β, cnt, ci, nb, hm, slot)
    return nothing
end

function fast_final1_kernel!(g1, g2, g3, g4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4,
                             y1, y2, y3, y4, mask, Na, tol, cnt, ci, hm, slot)
    β = blockIdx().x
    j = threadIdx().x
    nb = gridDim().x
    m = mask[β]
    z = zero(ComplexF64)
    n1 = z; n2 = z; n3 = z; n4 = z
    dsum = 0.0; gsum = 0.0
    if j <= Na
        @inbounds begin
            o1 = g1[j, β]; o2 = g2[j, β]; o3 = g3[j, β]; o4 = g4[j, β]
            n1 = _rk4_gnew(o1, a1[j, β], b1[j, β], c1[j, β], d1[j, β])
            n2 = _rk4_gnew(o2, a2[j, β], b2[j, β], c2[j, β], d2[j, β])
            n3 = _rk4_gnew(o3, a3[j, β], b3[j, β], c3[j, β], d3[j, β])
            n4 = _rk4_gnew(o4, a4[j, β], b4[j, β], c4[j, β], d4[j, β])
            dsum = abs2(n1 - o1) + abs2(n2 - o2) + abs2(n3 - o3) + abs2(n4 - o4)
            gsum = abs2(o1) + abs2(o2) + abs2(o3) + abs2(o4)
        end
    end
    sh = CuStaticSharedArray(Float64, (1024, 2))
    sh[j, 1] = dsum; sh[j, 2] = gsum
    nt = blockDim().x
    P = 1
    while P < nt
        P *= 2
    end
    if j <= P - nt
        sh[nt + j, 1] = 0.0; sh[nt + j, 2] = 0.0
    end
    sync_threads()
    s2 = P ÷ 2
    while s2 > 0
        if j <= s2
            sh[j, 1] += sh[j + s2, 1]; sh[j, 2] += sh[j + s2, 2]
        end
        sync_threads()
        s2 ÷= 2
    end
    done = sqrt(sh[1, 1]) / sqrt(sh[1, 2]) < tol
    if j <= Na
        @inbounds begin
            if m
                g1[j, β] = n1; g2[j, β] = n2; g3[j, β] = n3; g4[j, β] = n4
            end
            y1[j, β] = ComplexF32(g1[j, β]); y2[j, β] = ComplexF32(g2[j, β])
            y3[j, β] = ComplexF32(g3[j, β]); y4[j, β] = ComplexF32(g4[j, β])
        end
    end
    j == 1 && _fast_count_active!(m & !done, mask, β, cnt, ci, nb, hm, slot)
    return nothing
end

# one thread per channel: the new mask, and the count of channels still iterating after iteration ci
# (cnt[2ci - 1]) into the mapped slot once the last channel is in (cnt[2ci] counts the channels done)
@inline function _fast_count_active!(nm, mask, β, cnt, ci, nb, hm, slot)
    mask[β] = nm
    nm && CUDA.atomic_add!(pointer(cnt, 2ci - 1), Int32(1))
    CUDA.threadfence()
    t = CUDA.atomic_add!(pointer(cnt, 2ci), Int32(1))
    if t == nb - 1
        CUDA.threadfence()
        v = CUDA.atomic_add!(pointer(cnt, 2ci - 1), Int32(0))
        hm[slot] = Float64(v)
        CUDA.threadfence_system()
    end
    nothing
end

"""
    _fast_rk4!(g, meas_sq, model_sq, ne, Na, Nf, maxiter, tolerance, compact, ws) -> (converged, niters, fraction)

solve_gains!'s RK4 loop (full Jones, per channel, no prior / smoothing, tiled normal equations) in ~9
launches per iteration.  Gains as the default loop given the same per-channel stop decisions (the stop
sums run in another fixed order: rounding level).  The early stop is read one iteration late from
host-mapped memory without a stream sync; that extra iteration is a no-op (every channel frozen).
`converged`: every channel stopped (as the default loop), `fraction`: the fraction that did.
"""
function _fast_rk4!(g, meas_sq, model_sq, ne, Na, Nf, maxiter, tolerance, compact, ws)
    1 <= maxiter <= FAST_MAXITER || throw(ArgumentError("peel_fast: maxiter must be in 1:$FAST_MAXITER"))
    V = (meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy)
    M = (model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy)
    nbuf = Ref(0)
    buf4() = ntuple(_ -> _ws_buf!(ws, :solve, ComplexF64, (Na, Nf), true, nbuf[] += 1; zero=false), 4)
    k1, k2, k3, k4, x = (buf4() for _ in 1:5)
    y = ntuple(q -> _ws_buf!(ws, :fast_g32, ComplexF32, (Na, Nf), true, q; zero=false), 4)
    active = _ws_buf!(ws, :solve_active, Bool, (Nf,), true; zero=false)
    cnt = _ws_buf!(ws, :fast_cnt, Int32, (2 * maxiter,), true; zero=false)
    fill!(active, true); fill!(cnt, Int32(0))
    mp = _fast_mapped(); base = _fast_region()
    n = Na * Nf
    blocks, threads = kernel_config_1d(n)
    @cuda blocks=blocks threads=threads fast_tog32_kernel!(y..., g..., n)
    lowp = eltype(ne) == ComplexF32
    comp = compact === nothing ? nothing : compact.comp
    live = compact === nothing ? nothing : compact.live
    N = compact === nothing ? Na : compact.N
    ntile = cld(N, _NE_TILE)
    ty = NE_TY[]
    niters = 0; stopped = false
    for iter in 1:maxiter
        niters = iter
        for (stage, kk, gin) in ((1, k1, g), (2, k2, x), (3, k3, x), (4, k4, x))
            @cuda blocks=(ntile * (ntile + 1) ÷ 2, Nf) threads=(_NE_TILE, ty) shmem=8 * _NE_TILE * ty * sizeof(eltype(ne)) fast_ne_tile_kernel!(
                ne, y..., V..., M..., active, N, ntile, live)
            @cuda blocks=blocks threads=threads fast_step_kernel!(kk..., ne, gin..., g..., x..., y..., stage, active, Na, Nf,
                                                                 ntile, comp, lowp)
        end
        if Na <= 1024
            @cuda blocks=Nf threads=32 * cld(Na, 32) fast_final1_kernel!(g..., k1..., k2..., k3..., k4..., y..., active, Na,
                                                                         tolerance, cnt, iter, mp.d, base + iter)
        else
            @cuda blocks=Nf threads=_FAST_FT fast_final_kernel!(g..., k1..., k2..., k3..., k4..., y..., active, Na, tolerance,
                                                                cnt, iter, mp.d, base + iter)
        end
        CUDA.record(_fast_event(iter), CUDA.stream())
        if iter >= 2
            _fast_wait(_fast_event(iter - 1))
            if mp.h[base + iter - 1] == 0.0      # every channel stopped at iter - 1: iteration iter was a no-op
                niters = iter - 1; stopped = true
                break
            end
        end
    end
    stopped && return true, niters, 1.0
    _fast_wait(_fast_event(niters))
    nact = mp.h[base + niters]
    return nact == 0.0, niters, 1 - nact / Nf
end

"""
solve_gains! of a fast peel (rk4, :mixed, full Jones, no prior / smoothing, GPU, workspace): the square
matrices of `measured` into `meas_sq` (and of `model` unless `model_sq` is given), then `_fast_rk4!`.
"""
function _fast_solve!(cal::GPUCalibration, measured::GPUVisibilities, model::GPUVisibilities, meta::GPUMetadata,
                      model_sq, meas_sq, ws::PeelWorkspace, compact; maxiter::Integer, tolerance::Real, minuvw::Real)
    cal.is_diagonal && throw(ArgumentError("peel_fast: full-Jones calibrations only"))
    Na, Nf = meta.Nant, Nfreq(cal)
    Nfreq(measured) == Nf || throw(ArgumentError("peel_fast: per-channel calibrations only"))
    tolerance = max(Float64(tolerance), 1e-6)               # as solve_gains! with precision = :mixed
    minuvw = Float64(minuvw)
    _fast_square!(meas_sq, measured, meta, minuvw, compact)
    model_sq === nothing && throw(ArgumentError("peel_fast: the square model is built by the caller (cache_model_sq)"))
    ne = _ws_buf!(ws, :ne, ComplexF32, (Na, 8, cld(Na, _NE_TILE), Nf), true; zero=false)
    g = (cal.xx, cal.xy, cal.yx, cal.yy)
    _fast_rk4!(g, meas_sq, model_sq, ne, Na, Nf, Int(maxiter), tolerance, compact, ws)
end

#------------------------------------------------------------------------------#
# Residual guard                                                               #
#------------------------------------------------------------------------------#

const _GP_THREADS = 256
"""Number of power partial sums of `vis` (blocks of the power kernels)."""
_gp_blocks(vis) = cld(length(vis.xx), _GP_THREADS)

@inline _pw(a::ComplexF32, b::ComplexF32) = Float64(abs2(a) + abs2(b))
@inline _pw(a, b) = Float64(abs2(ComplexF64(a))) + Float64(abs2(ComplexF64(b)))
@inline _c32(x) = ComplexF32(x)

# corrupt_apply_kernel!'s update (mode 1 add, 2 subtract; mode 0 no update), then the power
# Σ |xx|² + |yy|² of the unflagged samples of the result per block of 256 threads (fixed tree) into `part`.
# L: the model × gain products in Float32 (with ComplexF32 gain copies)
function corrupt_apply_power_kernel!(part, dxx, dxy, dyx, dyy, flags, cxx, cxy, cyx, cyy, gxx, gxy, gyx, gyy,
                                     baselines, Nbase, Nfreq, Nfreq_cal, diag::Bool, mode::Int, ::Val{L}) where {L}
    tid = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tid
    p = 0.0
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        if mode != 0
            a1 = baselines[1, α]; a2 = baselines[2, α]
            b = Nfreq_cal == 1 ? 1 : β
            c0 = (cxx[α, β], cxy[α, β], cyx[α, β], cyy[α, β])
            c = L ? map(_c32, c0) : c0
            g1 = (gxx[a1, b], gxy[a1, b], gyx[a1, b], gyy[a1, b])
            g2 = (gxx[a2, b], gxy[a2, b], gyx[a2, b], gyy[a2, b])
            m = diag ? _corrupt_diag(c, g1, g2) : _corrupt_full(c, g1, g2)
            if mode == 2
                dxx[α, β] -= m[1]; dxy[α, β] -= m[2]; dyx[α, β] -= m[3]; dyy[α, β] -= m[4]
            else
                dxx[α, β] += m[1]; dxy[α, β] += m[2]; dyx[α, β] += m[3]; dyy[α, β] += m[4]
            end
        end
        p = flags[α, β] ? 0.0 : _pw(dxx[α, β], dyy[α, β])
    end
    sh = CuStaticSharedArray(Float64, _GP_THREADS)
    @inbounds sh[tid] = p
    sync_threads()
    s = _GP_THREADS ÷ 2
    while s > 0
        if tid <= s
            @inbounds sh[tid] += sh[tid + s]
        end
        sync_threads()
        s ÷= 2
    end
    tid == 1 && (@inbounds part[blockIdx().x] = sh[1])
    return nothing
end

# vis = (vis - C1) + C2 with C1 = J C Jᴴ of (c1, g1) and C2 of (c2, g2); part1: powers after the
# subtraction, part2: after the addition (the statements and block trees of two corrupt_apply_power_kernel! calls)
function subadd_power_kernel!(part1, part2, dxx, dxy, dyx, dyy, flags,
                              c1xx, c1xy, c1yx, c1yy, g1xx, g1xy, g1yx, g1yy,
                              c2xx, c2xy, c2yx, c2yy, g2xx, g2xy, g2yx, g2yy,
                              baselines, Nbase, Nfreq, nf1, nf2, diag1::Bool, diag2::Bool, ::Val{L}) where {L}
    tid = threadIdx().x
    idx = (blockIdx().x - 1) * blockDim().x + tid
    p1 = 0.0; p2 = 0.0
    if idx <= Nbase * Nfreq
        α = ((idx - 1) % Nbase) + 1
        β = ((idx - 1) ÷ Nbase) + 1
        a1 = baselines[1, α]; a2 = baselines[2, α]
        b = nf1 == 1 ? 1 : β
        c0 = (c1xx[α, β], c1xy[α, β], c1yx[α, β], c1yy[α, β])
        c = L ? map(_c32, c0) : c0
        ga = (g1xx[a1, b], g1xy[a1, b], g1yx[a1, b], g1yy[a1, b])
        gb = (g1xx[a2, b], g1xy[a2, b], g1yx[a2, b], g1yy[a2, b])
        m = diag1 ? _corrupt_diag(c, ga, gb) : _corrupt_full(c, ga, gb)
        x1 = dxx[α, β]; x2 = dxy[α, β]; x3 = dyx[α, β]; x4 = dyy[α, β]
        x1 -= m[1]; x2 -= m[2]; x3 -= m[3]; x4 -= m[4]
        fl = flags[α, β]
        p1 = fl ? 0.0 : _pw(x1, x4)
        b2 = nf2 == 1 ? 1 : β
        c0 = (c2xx[α, β], c2xy[α, β], c2yx[α, β], c2yy[α, β])
        c = L ? map(_c32, c0) : c0
        ga = (g2xx[a1, b2], g2xy[a1, b2], g2yx[a1, b2], g2yy[a1, b2])
        gb = (g2xx[a2, b2], g2xy[a2, b2], g2yx[a2, b2], g2yy[a2, b2])
        m = diag2 ? _corrupt_diag(c, ga, gb) : _corrupt_full(c, ga, gb)
        x1 += m[1]; x2 += m[2]; x3 += m[3]; x4 += m[4]
        dxx[α, β] = x1; dxy[α, β] = x2; dyx[α, β] = x3; dyy[α, β] = x4
        p2 = fl ? 0.0 : _pw(x1, x4)
    end
    sh = CuStaticSharedArray(Float64, _GP_THREADS)
    sh2 = CuStaticSharedArray(Float64, _GP_THREADS)
    @inbounds sh[tid] = p1
    @inbounds sh2[tid] = p2
    sync_threads()
    s = _GP_THREADS ÷ 2
    while s > 0
        if tid <= s
            @inbounds sh[tid] += sh[tid + s]
            @inbounds sh2[tid] += sh2[tid + s]
        end
        sync_threads()
        s ÷= 2
    end
    if tid == 1
        @inbounds part1[blockIdx().x] = sh[1]
        @inbounds part2[blockIdx().x] = sh2[1]
    end
    return nothing
end

"""
`vis ∓= J₁ C J₂ᴴ` (mode 2 / 1) or nothing (mode 0), and the power partial sums of the result into
`part`; the model × gain products in Float32 (ComplexF32 gain copies).  Not synchronised.
"""
function _apply_power!(part, vis, coherency, cal, meta, mode::Int)
    Nb, Nf = size(vis.xx)
    c = mode == 0 ? vis : coherency
    lowp = mode != 0
    g, nfc, dg = mode == 0 ? ((vis.xx, vis.xy, vis.yx, vis.yy), 1, false) :
                 (_g32!((cal.xx, cal.xy, cal.yx, cal.yy), 2), Nfreq(cal), cal.is_diagonal)
    @cuda blocks=_gp_blocks(vis) threads=_GP_THREADS corrupt_apply_power_kernel!(
        part, vis.xx, vis.xy, vis.yx, vis.yy, vis.flags, c.xx, c.xy, c.yx, c.yy, g...,
        meta.baselines, Nb, Nf, nfc, dg, mode, Val(lowp))
    part
end

# gain check partials: per block (finite flag, max |g|) over the 4 gain arrays
function fast_gaincheck_kernel!(pf, pm, g1, g2, g3, g4, n)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    fin = true; mx = 0.0
    while i <= n
        for a in (g1[i], g2[i], g3[i], g4[i])
            fin &= isfinite(a)
            mx = max(mx, abs(a))
        end
        i += blockDim().x * gridDim().x
    end
    sh = CuStaticSharedArray(Float64, 256)
    sf = CuStaticSharedArray(Int32, 256)
    sh[tid] = mx; sf[tid] = fin ? Int32(1) : Int32(0)
    sync_threads()
    s = 128
    while s > 0
        if tid <= s
            sh[tid] = max(sh[tid], sh[tid + s]); sf[tid] = min(sf[tid], sf[tid + s])
        end
        sync_threads()
        s ÷= 2
    end
    if tid == 1
        pm[blockIdx().x] = sh[1]; pf[blockIdx().x] = sf[1]
    end
    return nothing
end
function fast_gainfinal_kernel!(ok, pf, pm, nb, max_gain, hm, slot)
    if threadIdx().x == 1
        f = Int32(1); m = 0.0
        for b in 1:nb
            f = min(f, pf[b]); m = max(m, pm[b])
        end
        ok[1] = (f == 1 && m <= max_gain) ? Int32(1) : Int32(0)
        hm[slot] = Float64(f); hm[slot + 1] = m
        CUDA.threadfence_system()
    end
    return nothing
end
const _GCB = 64
function _fast_gaincheck!(ok, pf, pm, cal, max_gain, hm, slot)
    n = length(cal.xx)
    @cuda blocks=_GCB threads=256 fast_gaincheck_kernel!(pf, pm, cal.xx, cal.xy, cal.yx, cal.yy, n)
    @cuda threads=32 fast_gainfinal_kernel!(ok, pf, pm, _GCB, Float64(max_gain), hm, slot)
end

# power: one block, fixed order (strided sequential sums, then a tree) of `part` into hm[slot]
function fast_psum_kernel!(hm, slot, part, n)
    tid = threadIdx().x
    acc = 0.0
    i = tid
    while i <= n
        acc += part[i]; i += 1024
    end
    sh = CuStaticSharedArray(Float64, 1024)
    sh[tid] = acc
    sync_threads()
    s = 512
    while s > 0
        tid <= s && (sh[tid] += sh[tid + s])
        sync_threads()
        s ÷= 2
    end
    if tid == 1
        hm[slot] = sh[1]
        CUDA.threadfence_system()
    end
    return nothing
end
_fast_psum!(hm, slot, part) = @cuda threads=1024 fast_psum_kernel!(hm, slot, part, length(part))

# corrupt_apply_power_kernel! / subadd_power_kernel! that do nothing (no writes at all) unless ok[1] == 1
function fast_apply_power_kernel!(ok, part, dxx, dxy, dyx, dyy, flags, cxx, cxy, cyx, cyy, gxx, gxy, gyx, gyy,
                                  baselines, Nbase, Nfreq, Nfreq_cal, diag::Bool, mode::Int, ::Val{L}) where {L}
    ok[1] == Int32(1) || return nothing
    @inline corrupt_apply_power_kernel!(part, dxx, dxy, dyx, dyy, flags, cxx, cxy, cyx, cyy, gxx, gxy, gyx, gyy,
                                        baselines, Nbase, Nfreq, Nfreq_cal, diag, mode, Val(L))
    return nothing
end
function fast_subadd_power_kernel!(ok, part1, part2, dxx, dxy, dyx, dyy, flags,
                                   c1xx, c1xy, c1yx, c1yy, g1xx, g1xy, g1yx, g1yy,
                                   c2xx, c2xy, c2yx, c2yy, g2xx, g2xy, g2yx, g2yy,
                                   baselines, Nbase, Nfreq, nf1, nf2, diag1::Bool, diag2::Bool, ::Val{L}) where {L}
    ok[1] == Int32(1) || return nothing
    @inline subadd_power_kernel!(part1, part2, dxx, dxy, dyx, dyy, flags,
                                 c1xx, c1xy, c1yx, c1yy, g1xx, g1xy, g1yx, g1yy,
                                 c2xx, c2xy, c2yx, c2yy, g2xx, g2xy, g2yx, g2yy,
                                 baselines, Nbase, Nfreq, nf1, nf2, diag1, diag2, Val(L))
    return nothing
end
function _fast_apply_power!(ok, part, vis, coherency, cal, meta)
    Nb, Nf = size(vis.xx)
    g = _g32!((cal.xx, cal.xy, cal.yx, cal.yy), 2)
    @cuda blocks=_gp_blocks(vis) threads=_GP_THREADS fast_apply_power_kernel!(ok,
        part, vis.xx, vis.xy, vis.yx, vis.yy, vis.flags, coherency.xx, coherency.xy, coherency.yx, coherency.yy, g...,
        meta.baselines, Nb, Nf, Nfreq(cal), cal.is_diagonal, 2, Val(true))
end
function _fast_subadd_power!(ok, part1, part2, vis, c1, cl1, c2, cl2, meta)
    Nb, Nf = size(vis.xx)
    ga = _g32!((cl1.xx, cl1.xy, cl1.yx, cl1.yy), 3)
    gb = _g32!((cl2.xx, cl2.xy, cl2.yx, cl2.yy), 4)
    @cuda blocks=_gp_blocks(vis) threads=_GP_THREADS fast_subadd_power_kernel!(ok,
        part1, part2, vis.xx, vis.xy, vis.yx, vis.yy, vis.flags,
        c1.xx, c1.xy, c1.yx, c1.yy, ga...,
        c2.xx, c2.xy, c2.yx, c2.yy, gb...,
        meta.baselines, Nb, Nf, Nfreq(cl1), Nfreq(cl2), cl1.is_diagonal, cl2.is_diagonal, Val(true))
end

"""
    _fast_guarded_zest!(vis, meta, coherencies, calibrations, solve, peeliter, max_gain, report, ws)

`guarded_zest!` (require_convergence = false, a workspace, GPU) with one host wait per direction update:
the gain check, the subtraction (fused with the next direction's add-back when that direction is
subtracted) and the residual powers run on the device; a power is a fixed-order device sum of the
per-block partial sums (rounding level vs. residual_power).  Same control flow and accept / keep /
leave-in rules as guarded_zest!.
"""
function _fast_guarded_zest!(vis, meta, coherencies, calibrations, solve, peeliter, max_gain, report, ws)
    N = length(coherencies)
    all(c -> size(c.xx) == size(vis.xx), coherencies) ||
        throw(DimensionMismatch("every coherency must have the visibilities' size"))
    length(calibrations) == N || throw(ArgumentError("one calibration per coherency"))
    part = _ws_buf!(ws, :guard_power, Float64, (_gp_blocks(vis),), true; zero=false)
    part1 = _ws_buf!(ws, :fast_part1, Float64, (_gp_blocks(vis),), true; zero=false)
    okd = _ws_buf!(ws, :fast_ok, Int32, (1,), true; zero=false)
    pf = _ws_buf!(ws, :fast_pf, Int32, (_GCB,), true; zero=false)
    pm = _ws_buf!(ws, :fast_pm, Float64, (_GCB,), true; zero=false)
    mp = _fast_mapped()
    hm, dm = mp.h, mp.d
    # host-mapped slots: a region of 64 per wait cycle (every slot of a cycle is read after its wait)
    reg = Ref(_fast_region()); nxt = Ref(0)
    function slot!()
        nxt[] >= 60 && error("peel_fast: host-mapped slot region overflow")
        nxt[] += 1
        reg[] + nxt[]
    end
    newcycle!() = (reg[] = _fast_region(); nxt[] = 0)
    # `ver` counts the changes of vis; `part` holds the power partial sums of version `partver`
    ver = Ref(0); partver = Ref(-1)
    pver = Ref(-1); pval = Ref{Any}(0.0)     # power of version pver: a Float64, or a slot pending in this cycle
    function power()
        pver[] == ver[] && return pval[]
        partver[] == ver[] || _apply_power!(part, vis, nothing, nothing, meta, 0)
        partver[] = ver[]
        sl = slot!(); _fast_psum!(dm, sl, part)
        pver[] = ver[]; pval[] = sl
        sl
    end
    val(p::Float64) = p
    val(p::Int) = hm[p]
    sub!(c, cl) = (_apply_power!(part, vis, c, cl, meta, 2); ver[] += 1; partver[] = ver[]; nothing)
    add!(c, cl) = (_apply_power!(part, vis, c, cl, meta, 1); ver[] += 1; partver[] = ver[]; nothing)
    function wait!()
        ev = _fast_event(FAST_MAXITER + 1)
        CUDA.record(ev, CUDA.stream()); _fast_wait(ev)
        pval[] isa Int && (pval[] = hm[pval[]])     # resolve a pending power before its region is recycled
        nothing
    end
    subtracted = trues(N)
    for s in 1:N
        sub!(coherencies[s], calibrations[s])
    end
    pre = Ref{Any}(nothing)            # (iteration, s, p_old, p_with) of a direction added back by the fused pass
    function next_solve(it, k)
        i2, k2 = it, k + 1
        k2 > N && (i2 += 1; k2 = 1)
        i2 > peeliter && return nothing
        return (k2 == k || !subtracted[k2]) ? nothing : (i2, k2)
    end
    for iteration in 1:peeliter, s in 1:N
        coherency, cal = coherencies[s], calibrations[s]
        if pre[] !== nothing && pre[][1] == iteration && pre[][2] == s
            p_old, p_with = pre[][3], pre[][4]
            pre[] = nothing
        else
            pre[] === nothing || error("peel_fast: fused add-back of $(pre[]) not used at ($iteration, $s)")
            p_old = subtracted[s] ? power() : Inf
            subtracted[s] && add!(coherency, cal)
            p_with = power()
        end
        saved = _save_gains!(ws, cal)
        converged, niters = solve(cal, vis, coherency)
        gsl = slot!(); slot!(); nsl = slot!()      # gsl, gsl + 1: finite flag, max |g|
        _fast_gaincheck!(okd, pf, pm, cal, max_gain, dm, gsl)
        nx = next_solve(iteration, s)
        if nx === nothing
            _fast_apply_power!(okd, part, vis, coherency, cal, meta)
            _fast_psum!(dm, nsl, part)
        else
            nsl2 = slot!()
            _fast_subadd_power!(okd, part1, part, vis, coherency, cal, coherencies[nx[2]], calibrations[nx[2]], meta)
            _fast_psum!(dm, nsl, part1)
            _fast_psum!(dm, nsl2, part)
        end
        wait!()
        p_old = val(p_old); p_with = val(p_with)
        finite = hm[gsl] == 1.0; gmax = hm[gsl + 1]
        p_new = Inf
        if finite && gmax <= max_gain
            ver[] += 1; partver[] = ver[]
            if nx === nothing
                p_new = hm[nsl]; pver[] = ver[]; pval[] = p_new
            else
                p_new = hm[nsl]; p_next = hm[nsl2]; pver[] = ver[]; pval[] = p_next
                if p_new < p_with && p_new <= p_old
                    pre[] = (nx[1], nx[2], p_new, p_next)
                else
                    sub!(coherencies[nx[2]], calibrations[nx[2]])      # undo the fused add-back
                end
            end
            newcycle!()
            if p_new < p_with && p_new <= p_old
                subtracted[s] = true
                report === nothing || report(s, iteration; converged, niters, finite,
                                             max_gain=gmax, p_with, p_old, p_new, action=:accept)
                continue
            end
            add!(coherency, cal)       # undo the new subtraction
        else
            newcycle!()
        end
        _restore_gains!(cal, saved)
        if p_old < p_with
            sub!(coherency, cal)
            subtracted[s] = true
            action = :keep_previous
        else
            subtracted[s] = false
            action = :leave_in
        end
        report === nothing || report(s, iteration; converged, niters, finite,
                                     max_gain=gmax, p_with, p_old, p_new, action)
    end
    subtracted
end

#------------------------------------------------------------------------------#
# zest_sky_rfi!(...; peel_fast=true)                                           #
#------------------------------------------------------------------------------#

function _check_fast(vis, ws, kw)
    why = !_is_gpu(vis.xx) ? "GPU visibilities" :
          ws === nothing ? "a workspace (PeelWorkspace)" :
          !fast_kernels() ? "fast_kernels!(true)" :
          kw.solver != :rk4 ? "solver = :rk4" :
          kw.precision != :mixed ? "precision = :mixed" :
          kw.diagonal ? "diagonal = false" :
          kw.reg !== nothing ? "no phase prior (reg = nothing)" :
          (kw.smooth_flux != 0 || kw.rfi_smooth_flux != 0) ? "no smoothing (smooth_flux = rfi_smooth_flux = 0)" :
          !isempty(kw.restore) ? "no restored sources (restore = [])" :
          kw.rfi_mode != :after ? "rfi_mode = :after" :
          kw.initial_gains !== nothing ? "a cold start (initial_gains = nothing)" :
          !kw.cache_model_sq ? "cache_model_sq = true" :
          !(1 <= kw.maxiter <= FAST_MAXITER) ? "1 <= maxiter <= $FAST_MAXITER" :
          !(eltype(vis.xx) in (ComplexF32, ComplexF64)) ? "ComplexF32 or ComplexF64 visibilities" : ""
    isempty(why) || throw(ArgumentError("peel_fast needs $why"))
    nothing
end

"""
The fast peel: checks, the process-wide state, the peel of the rows with data (`zest_sky_rfi!` with
PEEL_FAST set) and the subtraction of the peeled directions on the fully flagged rows.
"""
function _zest_sky_rfi_fast!(vis, meta, mjd, sky, rfi, kw; flag_zeros::Bool, workspace)
    ws = workspace
    _check_fast(vis, ws, kw)
    Threads.atomic_cas!(_PEEL_FAST_BUSY, false, true) &&
        error("peel_fast: another fast peel is running in this process (one at a time)")
    try
        _fast_reset_caches!()
        PEEL_FAST[] = true
        isempty(rfi) || (_RFI_S0[] = uvw_geometry(meta)[1])
        return _zest_blc!(vis, meta, mjd, sky, rfi, kw, flag_zeros, ws)
    finally
        PEEL_FAST[] = false
        _fast_reset_caches!()
        _PEEL_FAST_BUSY[] = false
    end
end

# Baseline compaction: peel only the rows with an unflagged sample (padded with fully flagged zero rows to a
# multiple of _BLC_PAD, so the workspace shapes repeat), then subtract the peeled directions from the others.
function _zest_blc!(vis, meta, mjd, sky, rfi, kw, flag_zeros, ws)
    T = eltype(vis.xx)
    nzero = flag_zeros ? flag_zero_visibilities!(vis, ws) : 0
    alive = Array(vec(any(.!vis.flags; dims=2)))
    idx = findall(alive)
    n = length(idx)
    if n == 0                           # nothing to solve for: the rows as they are
        r = zest_sky_rfi!(vis, meta, mjd, sky, rfi; kw..., flag_zeros=false, workspace=ws, peel_fast=true)
        return merge(r, (flagged_zeros=nzero,))
    end
    meta_c, idx_d = _row_meta(meta, idx)
    npad = meta_c.Nbase
    # the inner call prepares the workspace with this key; preparing it here first keeps vc valid through it
    wkey = (meta.Nant, npad, meta.Nfreq, kw.precision, true)
    _ws_prepare!(ws, wkey)
    nr0 = ws.nreset
    vc = _ws_vis!(ws, :blc_vis, npad, meta.Nfreq, true, 1; zero=false, T=T)
    for f in (:xx, :xy, :yx, :yy)
        a = getfield(vc, f)
        view(a, 1:n, :) .= view(getfield(vis, f), idx_d, :)
        npad > n && (view(a, n+1:npad, :) .= zero(T))
    end
    view(vc.flags, 1:n, :) .= view(vis.flags, idx_d, :)
    npad > n && (view(vc.flags, n+1:npad, :) .= true)
    r = zest_sky_rfi!(vc, meta_c, mjd, sky, rfi; kw..., flag_zeros=false, workspace=ws, peel_fast=true)
    # a workspace reset inside would have freed vc
    (ws.key == wkey && ws.nreset == nr0) ||
        error("peel_fast: workspace key changed in the compacted peel ($(ws.key) vs $wkey, nreset $(ws.nreset) vs $nr0)")
    for f in (:xx, :xy, :yx, :yy)
        view(getfield(vis, f), idx_d, :) .= view(getfield(vc, f), 1:n, :)
    end
    _dead_subtract!(vis, meta, mjd, sky, rfi, r, findall(.!alive), ws)
    merge(r, (flagged_zeros=nzero,))
end

"""Metadata of the rows `idx` of `meta` padded to a multiple of _BLC_PAD rows (baseline (1, 1), UVW 0), and idx on the device."""
function _row_meta(meta, idx)
    n = length(idx)
    npad = cld(n, _BLC_PAD) * _BLC_PAD
    bl = Array(meta.baselines); uvw = Array(meta.uvw)
    blc = fill(Int32(1), 2, npad); blc[:, 1:n] .= bl[:, idx]
    uvc = zeros(Float64, 3, npad); uvc[:, 1:n] .= uvw[:, idx]
    GPUMetadata(meta.antenna_positions, CuArray(blc), meta.channels, meta.phase_center,
                meta.phase_center_ra, meta.phase_center_dec, CuArray(uvc)), CuArray(idx)
end

# The default subtracts every accepted direction on every row (the corrupt kernels ignore flags); the
# compacted peel left the fully flagged rows `dead` alone.  Each peeled direction (in the order of r.peeled)
# is generated on those rows, corrupted with its final gains and subtracted.
function _dead_subtract!(vis, meta, mjd, sky, rfi, r, dead, ws)
    (isempty(r.peeled) || isempty(dead)) && return nothing
    n = length(dead)
    meta_d, idx_d = _row_meta(meta, dead)
    npad = meta_d.Nbase; Nf = meta.Nfreq
    T = eltype(vis.xx)
    vd = _ws_vis!(ws, :dead_vis, npad, Nf, true, 1; zero=false, T=T)
    md = _ws_vis!(ws, :dead_model, npad, Nf, true, 1; zero=false, T=T)
    for f in (:xx, :xy, :yx, :yy)
        a = getfield(vd, f)
        view(a, 1:n, :) .= view(getfield(vis, f), idx_d, :)
        npad > n && (view(a, n+1:npad, :) .= zero(T))
    end
    vd.flags .= true
    part = _ws_buf!(ws, :dead_part, Float64, (_gp_blocks(vd),), true; zero=false)
    src = Dict(get_name(s) => s for s in vcat(collect(sky), collect(rfi)))
    for name in r.peeled
        s = src[name]; cal = r.gains.cals[name]
        if is_rfi(s)
            gpu_genvis!(md, meta_d, s, meta_d.phase_center_ra, meta_d.phase_center_dec, 0.0; overwrite=true,
                        s0=_RFI_S0[])
        else
            genvis_exact!(md, meta_d, s, mjd; overwrite=true)
        end
        _apply_power!(part, vd, md, cal, meta_d, 2)
    end
    for f in (:xx, :xy, :yx, :yy)
        view(getfield(vis, f), idx_d, :) .= view(getfield(vd, f), 1:n, :)
    end
    nothing
end
