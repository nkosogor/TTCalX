# StefCal solvers for peeling/zesting: CPU TTCal's RK4 solver, TTCalX's damped
# iteration, an optional Tukey phase prior, on GPU or host arrays.
#
# The per-(antenna, channel) arithmetic lives in @inline functions shared by
# the CUDA kernels and the host-thread loops, so both backends compute the
# same thing.

#==============================================================================#
#                         Short-baseline mask                                  #
#==============================================================================#

"""
    mask_short_baselines!(sq, meta, minuvw)

Zero every entry of the square visibilities whose baseline is shorter than
`minuvw` wavelengths: the result of `flag_short_baselines!`, computed with
broadcasts on the arrays' own device instead of a host round trip of all
four Nant×Nant×Nfreq matrices.
"""
function mask_short_baselines!(sq::GPUSquareVisibilities, meta::GPUMetadata, minuvw::Real)
    minuvw <= 0 && return sq
    pos = meta.antenna_positions
    L = sqrt.((pos[1, :] .- pos[1, :]').^2 .+ (pos[2, :] .- pos[2, :]').^2 .+ (pos[3, :] .- pos[3, :]').^2)
    Nf = size(sq.xx, 3)
    ch = meta.channels[1:min(Nf, length(meta.channels))]
    keep = reshape(L, size(L)..., 1) .>= reshape(minuvw .* (SPEED_OF_LIGHT ./ ch), 1, 1, :)
    for a in (sq.xx, sq.xy, sq.yx, sq.yy)
        a .= ifelse.(keep, a, zero(eltype(a)))
    end
    sq
end

"""
    square_visibilities!(sq, vis, meta, minuvw) -> sq

`gpu_makesquare!` followed by `mask_short_baselines!` in one kernel (GPU, per
channel): every baseline writes both of its entries, zero when flagged, an
autocorrelation or shorter than `minuvw` wavelengths.  When the baselines
cover every antenna pair the matrices need no zero fill first.  Not synchronised.
"""
function square_visibilities!(sq::GPUSquareVisibilities, vis::GPUVisibilities, meta::GPUMetadata,
                              minuvw::Real)
    Na, Nb, Nf = meta.Nant, meta.Nbase, Nfreq(vis)
    if !(sq.xx isa CuArray) || size(sq.xx, 3) != Nf
        gpu_makesquare!(sq, vis, meta)
        return mask_short_baselines!(sq, meta, minuvw)
    end
    if Nb != Na * (Na + 1) ÷ 2                         # some pairs never written
        for a in (sq.xx, sq.xy, sq.yx, sq.yy); fill!(a, zero(eltype(a))); end
    end
    blocks, threads = kernel_config_1d(Nb * Nf)
    @cuda blocks=blocks threads=threads makesquare_masked_kernel!(
        sq.xx, sq.xy, sq.yx, sq.yy, vis.xx, vis.xy, vis.yx, vis.yy, vis.flags,
        meta.baselines, meta.antenna_positions, meta.channels, Float64(minuvw), Nb, Nf)
    sq
end

function makesquare_masked_kernel!(sxx, sxy, syx, syy, vxx, vxy, vyx, vyy, flags,
                                   baselines, pos, channels, minuvw, Nb, Nf)
    idx = thread_index_1d()
    if idx <= Nb * Nf
        α = ((idx - 1) % Nb) + 1
        β = ((idx - 1) ÷ Nb) + 1
        a1 = baselines[1, α]; a2 = baselines[2, α]
        keep = !flags[α, β] && a1 != a2
        if keep && minuvw > 0
            L = sqrt((pos[1, a1] - pos[1, a2])^2 + (pos[2, a1] - pos[2, a2])^2 + (pos[3, a1] - pos[3, a2])^2)
            keep = L >= minuvw * (SPEED_OF_LIGHT / channels[β])
        end
        T = eltype(sxx)
        z = zero(T)
        xx = keep ? T(vxx[α, β]) : z; xy = keep ? T(vxy[α, β]) : z
        yx = keep ? T(vyx[α, β]) : z; yy = keep ? T(vyy[α, β]) : z
        sxx[a1, a2, β] = xx; sxy[a1, a2, β] = xy; syx[a1, a2, β] = yx; syy[a1, a2, β] = yy
        if a1 != a2
            sxx[a2, a1, β] = conj(xx); sxy[a2, a1, β] = conj(yx)
            syx[a2, a1, β] = conj(xy); syy[a2, a1, β] = conj(yy)
        end
    end
    return nothing
end

#==============================================================================#
#                         StefCal step (per element)                           #
#==============================================================================#

"""
Full-Jones normal equations for antenna j, channel β:
N = Σ_i (G_i M_ij)ᴴ V_ij and D = Σ_i (G_i M_ij)ᴴ (G_i M_ij).  The square
matrices are Hermitian (makesquare fills [a2, a1] with the conjugate
transpose of [a1, a2]), so M_ij is read as (M_ji)ᴴ: on the GPU neighbouring
threads j then read neighbouring addresses.
"""
@inline function _full_normal_equations(j, β, gxx, gxy, gyx, gyy,
                                        Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, Nant)
    T = eltype(Vxx)              # ComplexF32 for precision=:mixed: products and sums in single
    nxx = zero(T); nxy = zero(T); nyx = zero(T); nyy = zero(T)
    dxx = zero(T); dxy = zero(T); dyx = zero(T); dyy = zero(T)
    for i in 1:Nant
        gi_xx = T(gxx[i, β]); gi_xy = T(gxy[i, β]); gi_yx = T(gyx[i, β]); gi_yy = T(gyy[i, β])
        m_xx = conj(Mxx[j, i, β]); m_xy = conj(Myx[j, i, β])
        m_yx = conj(Mxy[j, i, β]); m_yy = conj(Myy[j, i, β])
        v_xx = conj(Vxx[j, i, β]); v_xy = conj(Vyx[j, i, β])
        v_yx = conj(Vxy[j, i, β]); v_yy = conj(Vyy[j, i, β])
        gm_xx = gi_xx*m_xx + gi_xy*m_yx
        gm_xy = gi_xx*m_xy + gi_xy*m_yy
        gm_yx = gi_yx*m_xx + gi_yy*m_yx
        gm_yy = gi_yx*m_xy + gi_yy*m_yy
        nxx += conj(gm_xx)*v_xx + conj(gm_yx)*v_yx
        nxy += conj(gm_xx)*v_xy + conj(gm_yx)*v_yy
        nyx += conj(gm_xy)*v_xx + conj(gm_yy)*v_yx
        nyy += conj(gm_xy)*v_xy + conj(gm_yy)*v_yy
        dxx += conj(gm_xx)*gm_xx + conj(gm_yx)*gm_yx
        dxy += conj(gm_xx)*gm_xy + conj(gm_yx)*gm_yy
        dyx += conj(gm_xy)*gm_xx + conj(gm_yy)*gm_yx
        dyy += conj(gm_xy)*gm_xy + conj(gm_yy)*gm_yy
    end
    map(ComplexF64, (nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy))
end

"""
Full-Jones StefCal step for antenna j, channel β: (D⁻¹N)ᴴ - G_j, or zero
when D is singular.  With a prior (`start !== nothing`) each diagonal term
of X = G_jᴴ is regularized on the Schur complement of its column's quadratic
cost, and the off-diagonal term is moved to its conditional optimum.
"""
@inline _full_step(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                   start, width, lam, Nant) =
    _full_step_ne(j, β, _full_normal_equations(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy,
                                               Mxx, Mxy, Myx, Myy, Nant),
                  gxx, gxy, gyx, gyy, start, width, lam)

"""StefCal step of antenna j, channel β from its normal equations `ne` = (N, D) (see `_full_step`)."""
@inline function _full_step_ne(j, β, ne, gxx, gxy, gyx, gyy, start, width, lam)
    nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy = ne
    z = ComplexF64(0)
    det_den = dxx*dyy - dxy*dyx
    abs(det_den) > eps(Float64) || return z, z, z, z
    inv_det = 1.0 / det_den
    if start === nothing
        inv_xx = dyy * inv_det; inv_xy = -dxy * inv_det
        inv_yx = -dyx * inv_det; inv_yy = dxx * inv_det
        sol_xx = inv_xx*nxx + inv_xy*nyx
        sol_xy = inv_xx*nxy + inv_xy*nyy
        sol_yx = inv_yx*nxx + inv_yy*nyx
        sol_yy = inv_yx*nxy + inv_yy*nyy
        return conj(sol_xx) - gxx[j, β], conj(sol_yx) - gxy[j, β],
               conj(sol_xy) - gyx[j, β], conj(sol_yy) - gyy[j, β]
    end
    x11 = ( dyy*nxx - dxy*nyx) * inv_det
    x12 = ( dyy*nxy - dxy*nyy) * inv_det
    x21 = (-dyx*nxx + dxx*nyx) * inv_det
    x22 = (-dyx*nxy + dxx*nyy) * inv_det
    st = start[j, β]; wd = width[j, β]
    d11 = real(dxx); d22 = real(dyy); d12sq = abs2(dxy)
    if d22 > 0.0        # column 1 = (conj G_xx, conj G_xy): regularize G_xx
        zz = regularize_element(conj(x11), d11 - d12sq / d22, st, wd, lam)
        δ1 = conj(zz) - x11
        x11 += δ1
        x21 += -dyx * δ1 / d22
    end
    if d11 > 0.0        # column 2 = (conj G_yx, conj G_yy): regularize G_yy
        zz = regularize_element(conj(x22), d22 - d12sq / d11, st, wd, lam)
        δ2 = conj(zz) - x22
        x22 += δ2
        x12 += -dxy * δ2 / d11
    end
    conj(x11) - gxx[j, β], conj(x21) - gxy[j, β], conj(x12) - gyx[j, β], conj(x22) - gyy[j, β]
end

"""Diagonal normal equations of antenna j, channel β: (nxx, nyy, dxx, dyy)."""
@inline function _diag_normal_equations(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, Nant)
    T = eltype(Vxx)
    nxx = zero(T); nyy = zero(T); dxx = zero(T); dyy = zero(T)
    for i in 1:Nant
        gi_xx = T(gxx[i, β]); gi_yy = T(gyy[i, β])
        gm_xx = gi_xx * Mxx[i, j, β]; gm_xy = gi_xx * Mxy[i, j, β]
        gm_yx = gi_yy * Myx[i, j, β]; gm_yy = gi_yy * Myy[i, j, β]
        nxx += conj(gm_xx) * Vxx[i, j, β] + conj(gm_yx) * Vyx[i, j, β]
        nyy += conj(gm_xy) * Vxy[i, j, β] + conj(gm_yy) * Vyy[i, j, β]
        dxx += conj(gm_xx) * gm_xx + conj(gm_yx) * gm_yx
        dyy += conj(gm_xy) * gm_xy + conj(gm_yy) * gm_yy
    end
    map(ComplexF64, (nxx, nyy, dxx, dyy))
end

"""Diagonal StefCal step for antenna j, channel β (as `stefcal_step_diagonal_kernel!`), optionally regularized."""
@inline function _diag_step(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
    nxx, nyy, dxx, dyy = _diag_normal_equations(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, Nant)
    new_xx = abs(dxx) > eps(Float64) ? conj(nxx / dxx) : gxx[j, β]
    new_yy = abs(dyy) > eps(Float64) ? conj(nyy / dyy) : gyy[j, β]
    if start !== nothing
        new_xx = regularize_element(new_xx, real(dxx), start[j, β], width[j, β], lam)
        new_yy = regularize_element(new_yy, real(dyy), start[j, β], width[j, β], lam)
    end
    new_xx - gxx[j, β], new_yy - gyy[j, β]
end

function stefcal_full_step_kernel!(kxx, kxy, kyx, kyy, gxx, gxy, gyx, gyy,
                                   Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                                   start, width, lam, active, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        if active !== nothing && !active[β]            # frozen channel: no step
            z = zero(eltype(kxx))
            kxx[j, β] = z; kxy[j, β] = z; kyx[j, β] = z; kyy[j, β] = z
            return nothing
        end
        sxx, sxy, syx, syy = _full_step(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy,
                                        Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
        kxx[j, β] = sxx; kxy[j, β] = sxy; kyx[j, β] = syx; kyy[j, β] = syy
    end
    return nothing
end

function stefcal_diag_step_kernel!(kxx, kyy, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                                   start, width, lam, active, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        if active !== nothing && !active[β]
            kxx[j, β] = zero(eltype(kxx)); kyy[j, β] = zero(eltype(kyy))
            return nothing
        end
        sxx, syy = _diag_step(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
        kxx[j, β] = sxx; kyy[j, β] = syy
    end
    return nothing
end

#==============================================================================#
#      Full-Jones normal equations from the upper triangle (GPU, tiled)        #
#==============================================================================#
#
# The square matrices are Hermitian, so each baseline (r, c) feeds two antennas:
# c with i = r (block [r, c], gain G_r) and r with i = c (block [c, r] = [r, c]ᴴ,
# gain G_c).  A CUDA block takes one 32×32 tile (I ≤ J) of channel β and reads it
# once: half the bytes of the per-antenna loop, which is memory bound in single
# precision.  Partial sums go to `part[j, q, slot, β]` (slot = the other tile),
# so the channel sums are deterministic; `full_step_ne_kernel!` adds them up.

const _NE_TILE = 32
const NE_TY = Ref(2)            # threads per block (32, NE_TY[]); each thread takes 32/NE_TY[] columns

@inline function _ne_terms(g, m, v)
    gm_xx = g[1]*m[1] + g[2]*m[3]
    gm_xy = g[1]*m[2] + g[2]*m[4]
    gm_yx = g[3]*m[1] + g[4]*m[3]
    gm_yy = g[3]*m[2] + g[4]*m[4]
    (conj(gm_xx)*v[1] + conj(gm_yx)*v[3], conj(gm_xx)*v[2] + conj(gm_yx)*v[4],
     conj(gm_xy)*v[1] + conj(gm_yy)*v[3], conj(gm_xy)*v[2] + conj(gm_yy)*v[4],
     conj(gm_xx)*gm_xx + conj(gm_yx)*gm_yx, conj(gm_xx)*gm_xy + conj(gm_yx)*gm_yy,
     conj(gm_xy)*gm_xx + conj(gm_yy)*gm_yx, conj(gm_xy)*gm_xy + conj(gm_yy)*gm_yy)
end

@inline function _warp_sum(x::Complex)
    re = real(x); im = imag(x)
    off = 16
    while off > 0
        re += CUDA.shfl_down_sync(0xffffffff, re, off)
        im += CUDA.shfl_down_sync(0xffffffff, im, off)
        off >>= 1
    end
    complex(re, im)
end

@inline _hblock(a) = (conj(a[1]), conj(a[3]), conj(a[2]), conj(a[4]))

function full_ne_tile_kernel!(part, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                              active, Nant, ntile)
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
    gr = r <= Nant ? (T(gxx[r, β]), T(gxy[r, β]), T(gyx[r, β]), T(gyy[r, β])) : (zt, zt, zt, zt)
    racc = z8
    for k in 0:(_NE_TILE ÷ TY - 1)
        c = J * _NE_TILE + ty + TY * k
        ccon = z8
        if r <= Nant && c <= Nant
            m = (Mxx[r, c, β], Mxy[r, c, β], Myx[r, c, β], Myy[r, c, β])
            v = (Vxx[r, c, β], Vxy[r, c, β], Vyx[r, c, β], Vyy[r, c, β])
            ccon = _ne_terms(gr, m, v)                  # antenna c, i = r
            if I != J                                   # antenna r, i = c
                gc = (T(gxx[c, β]), T(gxy[c, β]), T(gyx[c, β]), T(gyy[c, β]))
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

"""Normal equations (N, D) of antenna j, channel β from the tile partial sums (fixed order)."""
@inline function _sum_part(part, j, β, ntile)
    ntuple(Val(8)) do q
        acc = part[j, q, 1, β]
        for t in 2:ntile
            acc += part[j, q, t, β]
        end
        ComplexF64(acc)
    end
end

function _launch_ne_tiles!(part, g, V, M, active, Nant, Nf)
    ntile = cld(Nant, _NE_TILE)
    ty = NE_TY[]
    @cuda blocks=(ntile * (ntile + 1) ÷ 2, Nf) threads=(_NE_TILE, ty) shmem=8 * _NE_TILE * ty * sizeof(eltype(part)) full_ne_tile_kernel!(
        part, g..., V..., M..., active, Nant, ntile)
    ntile
end

function full_step_ne_kernel!(kxx, kxy, kyx, kyy, part, gxx, gxy, gyx, gyy,
                              start, width, lam, active, Nant, Nfreq, ntile)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        if active !== nothing && !active[β]
            z = zero(eltype(kxx))
            kxx[j, β] = z; kxy[j, β] = z; kyx[j, β] = z; kyy[j, β] = z
            return nothing
        end
        ne = _sum_part(part, j, β, ntile)
        sxx, sxy, syx, syy = _full_step_ne(j, β, ne, gxx, gxy, gyx, gyy, start, width, lam)
        kxx[j, β] = sxx; kxy[j, β] = sxy; kyx[j, β] = syx; kyy[j, β] = syy
    end
    return nothing
end

"""Scratch for the tiled normal equations of `Nant` antennas, `Nf` channels."""
ne_scratch(::Type{T}, Nant::Int, Nf::Int) where {T} = CUDA.zeros(T, Nant, 8, cld(Nant, _NE_TILE), Nf)

function _tiled_full_steps!(k, g, V, M, start, width, lam, active, Nant, Nf, part)
    ntile = _launch_ne_tiles!(part, g, V, M, active, Nant, Nf)
    blocks, threads = kernel_config_1d(Nant * Nf)
    @cuda blocks=blocks threads=threads full_step_ne_kernel!(
        k..., part, g..., start, width, lam, active, Nant, Nf, ntile)
    k
end

const TILED_STEPS = Ref(true)          # false: the per-antenna loop kernel (for comparisons)

"""
    stefcal_steps!(k, g, is_diagonal, meas_sq, model_sq, Nant; reg=nothing, active=nothing)

StefCal steps k = G_new(g) - g for all antennas and channels, for gains
`g = (xx, xy, yx, yy)` (`xy`, `yx` unused when diagonal), on the arrays'
device.  `reg` is an optional `PhaseRegularization`.  `active` (a Bool vector
per channel on the gains' device) skips frozen channels: their step is zero.
The square visibilities may be ComplexF32 (normal equations in single precision).
GPU launches are not synchronised: later work on the stream is ordered after them.
"""
function stefcal_steps!(k, g, is_diagonal::Bool, meas_sq::GPUSquareVisibilities,
                        model_sq::GPUSquareVisibilities, Nant::Int; reg=nothing, active=nothing,
                        ne=nothing)
    Nf = size(g[1], 2)
    start, width, lam = reg === nothing ? (nothing, nothing, 0.0) : (reg.start, reg.width, reg.lambda)
    V = (meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy)
    M = (model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy)
    if g[1] isa CuArray
        blocks, threads = kernel_config_1d(Nant * Nf)
        if !is_diagonal && TILED_STEPS[]
            part = ne === nothing ? ne_scratch(eltype(V[1]), Nant, Nf) : ne
            _tiled_full_steps!(k, g, V, M, start, width, lam, active, Nant, Nf, part)
        elseif is_diagonal
            @cuda blocks=blocks threads=threads stefcal_diag_step_kernel!(
                k[1], k[4], g[1], g[4], V..., M..., start, width, lam, active, Nant, Nf)
        else
            @cuda blocks=blocks threads=threads stefcal_full_step_kernel!(
                k..., g..., V..., M..., start, width, lam, active, Nant, Nf)
        end
    else
        Threads.@threads for β in 1:Nf
            if active !== nothing && !active[β]
                for c in (is_diagonal ? (1, 4) : (1, 2, 3, 4)); k[c][:, β] .= 0; end
                continue
            end
            @inbounds for j in 1:Nant
                if is_diagonal
                    k[1][j, β], k[4][j, β] = _diag_step(j, β, g[1], g[4], V..., M..., start, width, lam, Nant)
                else
                    k[1][j, β], k[2][j, β], k[3][j, β], k[4][j, β] =
                        _full_step(j, β, g..., V..., M..., start, width, lam, Nant)
                end
            end
        end
    end
    k
end

#==============================================================================#
#            Consensus step: per-channel gains tied to one band gain           #
#==============================================================================#
#
# Per antenna j the linearised StefCal cost of channel c is
# (X_c - X̂_c)ᴴ D_c (X_c - X̂_c) with X̂_c = D_c⁻¹ N_c (X = G_jᴴ).  Adding
# ρ Σ_c ‖X_c - X̄‖² with one shared X̄ and minimising over both gives, with
# A_c = (D_c + ρ)⁻¹,
#     X̄ = (Σ_c A_c D_c)⁻¹ Σ_c A_c N_c,    X_c = A_c (N_c + ρ X̄).
# ρ → ∞ is one gain per band (X̄ = (ΣD)⁻¹ ΣN), ρ → 0 the per-channel solve.
# A channel with D_c ≫ ρ keeps its own gain; one with D_c ≪ ρ gets X̄, and so
# does a flagged channel (D_c = 0): no D_c⁻¹ is needed.  2×2 matrices are
# tuples (xx, xy, yx, yy).

@inline _m2mul(a, b) = (a[1]*b[1] + a[2]*b[3], a[1]*b[2] + a[2]*b[4],
                        a[3]*b[1] + a[4]*b[3], a[3]*b[2] + a[4]*b[4])

"""(ok, a⁻¹) of a 2×2 matrix; ok = false when det is negligible relative to its scale."""
@inline function _m2inv(a)
    det = a[1]*a[4] - a[2]*a[3]
    scale = abs(a[1]) + abs(a[2]) + abs(a[3]) + abs(a[4])
    ok = scale > 0 && abs(det) > 1e-13 * scale * scale
    d = ok ? det : one(det)
    ok, (a[4]/d, -a[2]/d, -a[3]/d, a[1]/d)
end

"""Store A = (D + ρ)⁻¹, A N and A D of antenna j, channel β in the 12 scratch arrays `w`."""
@inline function _consensus_prep!(w, j, β, g, V, M, rho, is_diagonal, Nant)
    if is_diagonal
        nxx, nyy, dxx, dyy = _diag_normal_equations(j, β, g[1], g[4], V..., M..., Nant)
        ax = 1.0 / (dxx + rho); ay = 1.0 / (dyy + rho)
        z = zero(ax)
        a = (ax, z, z, ay); an = (ax * nxx, z, z, ay * nyy); ad = (ax * dxx, z, z, ay * dyy)
        for q in 1:4
            w[q][j, β] = a[q]; w[4 + q][j, β] = an[q]; w[8 + q][j, β] = ad[q]
        end
        return nothing
    end
    _consensus_prep_ne!(w, j, β, _full_normal_equations(j, β, g..., V..., M..., Nant), rho)
end

"""Full-Jones `_consensus_prep!` from the normal equations `ne` of antenna j, channel β."""
@inline function _consensus_prep_ne!(w, j, β, ne, rho)
    nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy = ne
    _, a = _m2inv((dxx + rho, dxy, dyx, dyy + rho))     # D + ρ is positive definite
    an = _m2mul(a, (nxx, nxy, nyx, nyy))
    ad = _m2mul(a, (dxx, dxy, dyx, dyy))
    for q in 1:4
        w[q][j, β] = a[q]; w[4 + q][j, β] = an[q]; w[8 + q][j, β] = ad[q]
    end
    nothing
end

function consensus_prep_ne_kernel!(w, part, rho, Nant, Nfreq, ntile)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1; β = ((idx - 1) ÷ Nant) + 1
        _consensus_prep_ne!(w, j, β, _sum_part(part, j, β, ntile), rho)
    end
    return nothing
end

"""Band gain X̄ of antenna j from the channel sums; per polarisation when diagonal."""
@inline function _consensus_band_gain(san, sad, j, is_diagonal)
    if is_diagonal
        okx = abs(sad[1][j]) > 0; oky = abs(sad[4][j]) > 0
        z = zero(ComplexF64)
        return (okx, oky), (okx ? san[1][j] / sad[1][j] : z, z, z, oky ? san[4][j] / sad[4][j] : z)
    end
    ok, inv_sad = _m2inv((sad[1][j], sad[2][j], sad[3][j], sad[4][j]))
    (ok, ok), _m2mul(inv_sad, (san[1][j], san[2][j], san[3][j], san[4][j]))
end

"""Step of antenna j, channel β from the scratch `w` and the channel sums `san`, `sad`."""
@inline function _consensus_finish!(k, g, w, san, sad, j, β, rho, is_diagonal)
    (okx, oky), xbar = _consensus_band_gain(san, sad, j, is_diagonal)
    ax = _m2mul((w[1][j, β], w[2][j, β], w[3][j, β], w[4][j, β]), xbar)
    x = ntuple(q -> w[4 + q][j, β] + rho * ax[q], 4)
    z = zero(ComplexF64)
    # G = Xᴴ; a polarisation (or antenna) without data keeps its gain
    k[1][j, β] = okx ? conj(x[1]) - g[1][j, β] : z
    k[4][j, β] = oky ? conj(x[4]) - g[4][j, β] : z
    if !is_diagonal
        k[2][j, β] = okx ? conj(x[3]) - g[2][j, β] : z
        k[3][j, β] = okx ? conj(x[2]) - g[3][j, β] : z
    end
    nothing
end

function consensus_prep_kernel!(w, g, V, M, rho, is_diagonal, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        _consensus_prep!(w, ((idx - 1) % Nant) + 1, ((idx - 1) ÷ Nant) + 1, g, V, M, rho, is_diagonal, Nant)
    end
    return nothing
end

function consensus_finish_kernel!(k, g, w, san, sad, rho, is_diagonal, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        _consensus_finish!(k, g, w, san, sad, ((idx - 1) % Nant) + 1, ((idx - 1) ÷ Nant) + 1, rho, is_diagonal)
    end
    return nothing
end

"""
    consensus_steps!(k, g, is_diagonal, meas_sq, model_sq, Nant, rho, w)

StefCal steps k = G_new(g) - g in which every channel's gain is tied to one
band gain with weight `rho` (see the comment above).  `w` is 12 Nant×Nfreq
scratch arrays on the gains' device.
"""
function consensus_steps!(k, g, is_diagonal::Bool, meas_sq::GPUSquareVisibilities,
                          model_sq::GPUSquareVisibilities, Nant::Int, rho::Float64, w; ne=nothing)
    Nf = size(g[1], 2)
    V = (meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy)
    M = (model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy)
    sums() = (ntuple(q -> vec(sum(w[4 + q]; dims=2)), 4), ntuple(q -> vec(sum(w[8 + q]; dims=2)), 4))
    if g[1] isa CuArray
        blocks, threads = kernel_config_1d(Nant * Nf)
        if !is_diagonal && TILED_STEPS[]
            part = ne === nothing ? ne_scratch(eltype(V[1]), Nant, Nf) : ne
            ntile = _launch_ne_tiles!(part, g, V, M, nothing, Nant, Nf)
            @cuda blocks=blocks threads=threads consensus_prep_ne_kernel!(w, part, rho, Nant, Nf, ntile)
        else
            @cuda blocks=blocks threads=threads consensus_prep_kernel!(w, g, V, M, rho, is_diagonal, Nant, Nf)
        end
        san, sad = sums()
        @cuda blocks=blocks threads=threads consensus_finish_kernel!(k, g, w, san, sad, rho, is_diagonal, Nant, Nf)
    else
        Threads.@threads for β in 1:Nf
            @inbounds for j in 1:Nant
                _consensus_prep!(w, j, β, g, V, M, rho, is_diagonal, Nant)
            end
        end
        san, sad = sums()
        Threads.@threads for β in 1:Nf
            @inbounds for j in 1:Nant
                _consensus_finish!(k, g, w, san, sad, j, β, rho, is_diagonal)
            end
        end
    end
    k
end

#==============================================================================#
#                                 Solver                                       #
#==============================================================================#

"""
    solve_gains!(cal, measured, model, meta; solver=:rk4, maxiter=5, tolerance=1e-4,
                 minuvw=0.0, reg=nothing) -> (converged, niters, fraction_converged)

Solve for the gains `cal` (updated in place) that best fit `measured` with
`model`, as in one direction of a peel/zest.

- `solver=:rk4` is CPU TTCal's `solve!`: each iteration takes an RK4 step of
  the StefCal update, and each channel stops separately once
  ‖ΔG‖/‖G‖ < `tolerance` (or after `maxiter` iterations).
- `solver=:damped` is `gpu_stefcal!`'s iteration: half steps and one global
  test ‖step‖ < `tolerance`·‖G‖ over all channels.

Baselines shorter than `minuvw` wavelengths are excluded from the solve, as
are flagged visibilities (set the model's flags to the data's to exclude
them from both sides).  `reg` is an optional `PhaseRegularization`.

`smooth_flux = S > 0` (Jy) ties every channel's gain to one band gain with
weight ρ = Nant (S/2)², the normal-equation weight of a direction of apparent
flux S (`consensus_steps!`): a direction much brighter than S keeps
per-channel gains, a much fainter one gets the band gain, which still fits its
own model but cannot follow another source's fringe from channel to channel.
The channels are then coupled, so the RK4 solver stops them together.  Not
combined with `reg` or wideband calibrations.

`precision = :mixed` stores the square visibilities as ComplexF32 and sums the
normal equations in single precision (gains, the 2×2 solves and the
convergence test stay double): on GPUs with slow FP64 (RTX A4000: 1/32 rate)
the step kernel, most of the solve, gets several times faster.  Gains agree
with `:double` to ~1e-7, the step size then stalls near 1e-7, so `tolerance`
is raised to at least 1e-6.  Wideband calibrations always use double.

Works on GPU arrays and, per channel only, on host arrays (one channel per
thread).  `converged` is true when every channel converged.
"""
function solve_gains!(calibration::GPUCalibration, measured::GPUVisibilities,
                      model::GPUVisibilities, meta::GPUMetadata;
                      solver::Symbol=:rk4, maxiter::Integer=5, tolerance::Real=1e-4,
                      minuvw::Real=0.0, reg=nothing, smooth_flux::Real=0.0,
                      precision::Symbol=:double, model_sq=nothing, meas_sq=nothing)
    solver in (:rk4, :damped) || throw(ArgumentError("unknown solver $solver (use :rk4 or :damped)"))
    tolerance, minuvw = Float64(tolerance), Float64(minuvw)
    use_gpu = _is_gpu(measured)
    Na = meta.Nant
    wideband = Nfreq(calibration) == 1 && Nfreq(measured) > 1
    !use_gpu && wideband && throw(ArgumentError("host solve_gains! supports per-channel calibrations only"))
    precision in (:double, :mixed) || throw(ArgumentError("precision must be :double or :mixed"))
    precision == :mixed && (tolerance = max(tolerance, 1e-6))     # single-precision floor ~1e-7
    smooth_flux >= 0 || throw(ArgumentError("smooth_flux must be ≥ 0 (got $smooth_flux)"))
    smooth_flux > 0 && (reg !== nothing || wideband) &&
        throw(ArgumentError("smooth_flux cannot be combined with reg or a wideband calibration"))
    if reg !== nothing
        size(reg.start) == (Na, Nfreq(calibration)) || throw(ArgumentError(
            "phase prior is $(size(reg.start)), calibration is $((Na, Nfreq(calibration))) " *
            "(use phase_regularization(...; nfreq=1) for wideband calibrations)"))
        (reg.start isa CuArray) == _is_gpu(calibration) || throw(ArgumentError(
            "phase prior and calibration must be on the same device (phase_regularization(...; gpu=...))"))
    end
    Tsq = precision == :mixed && !wideband ? ComplexF32 : ComplexF64
    nsq = wideband ? 1 : Nfreq(measured)
    # `meas_sq`: optional reused scratch; `model_sq`: optional precomputed square model
    # (square_model(model, meta; minuvw, precision), constant across peel passes)
    meas_sq = meas_sq !== nothing && eltype(meas_sq.xx) == Tsq && size(meas_sq.xx, 3) == nsq &&
              _is_gpu(meas_sq.xx) == use_gpu ? meas_sq : GPUSquareVisibilities(Na, nsq, gpu=use_gpu, T=Tsq)
    if use_gpu
        square_visibilities!(meas_sq, measured, meta, minuvw)
        model_sq = model_sq !== nothing ? model_sq :
            square_visibilities!(GPUSquareVisibilities(Na, nsq, gpu=true, T=Tsq), model, meta, minuvw)
    else
        host_makesquare!(meas_sq, measured, meta)
        mask_short_baselines!(meas_sq, meta, minuvw)
        if model_sq === nothing
            model_sq = GPUSquareVisibilities(Na, nsq, gpu=false, T=Tsq)
            host_makesquare!(model_sq, model, meta)
            mask_short_baselines!(model_sq, meta, minuvw)
        end
    end
    eltype(model_sq.xx) == Tsq && size(model_sq.xx, 3) == nsq ||
        throw(ArgumentError("model_sq does not match precision / channels of this solve"))
    ne = use_gpu && !calibration.is_diagonal ? ne_scratch(Tsq, Na, nsq) : nothing

    Nf = Nfreq(calibration)
    zero_like() = ntuple(_ -> use_gpu ? CUDA.zeros(ComplexF64, Na, Nf) : zeros(ComplexF64, Na, Nf), 4)
    g = (calibration.xx, calibration.xy, calibration.yx, calibration.yy)
    diag = calibration.is_diagonal
    comps = diag ? (1, 4) : (1, 2, 3, 4)
    rho = Na * (Float64(smooth_flux) / 2)^2
    w = rho > 0 ? ntuple(_ -> use_gpu ? CUDA.zeros(ComplexF64, Na, Nf) : zeros(ComplexF64, Na, Nf), 12) : nothing
    act = Ref{Any}(nothing)               # RK4: channels still iterating (GPU or host Bool vector)
    steps!(k, x) = rho > 0 ? consensus_steps!(k, x, diag, meas_sq, model_sq, Na, rho, w; ne=ne) :
        stefcal_steps!(k, x, diag, meas_sq, model_sq, Na; reg=reg, active=act[], ne=ne)

    if solver == :damped
        k = zero_like()
        converged = false
        niters = 0
        for iter in 1:maxiter
            niters = iter
            steps!(k, g)
            norm_step = sqrt(sum(sum(abs2, k[c]) for c in comps))
            norm_gain = sqrt(sum(sum(abs2, g[c]) for c in comps))
            converged = norm_step < tolerance * norm_gain
            for c in comps
                g[c] .+= 0.5 .* k[c]
            end
            converged && break
        end
        return converged, niters, converged ? 1.0 : 0.0
    end

    k1, k2, k3, k4, x, gnew = (zero_like() for _ in 1:6)
    active = trues(Nf)
    niters = 0
    for iter in 1:maxiter
        any(active) || break
        niters = iter
        act[] = all(active) ? nothing : (use_gpu ? CuArray(active) : copy(active))
        steps!(k1, g)
        for c in comps; x[c] .= g[c] .+ 0.5 .* k1[c]; end
        steps!(k2, x)
        for c in comps; x[c] .= g[c] .+ 0.5 .* k2[c]; end
        steps!(k3, x)
        for c in comps; x[c] .= g[c] .+ k3[c]; end
        steps!(k4, x)
        for c in comps
            gnew[c] .= g[c] .+ (k1[c] .+ 2 .* k2[c] .+ 2 .* k3[c] .+ k4[c]) ./ 6
        end
        # ‖ΔG‖² and ‖G‖² per channel, summed on the device: one transfer per iteration
        dg = sum(sum(abs2.(gnew[c] .- g[c]); dims=1) for c in comps)
        gg = sum(sum(abs2.(g[c]); dims=1) for c in comps)
        both = Array(vcat(dg, gg))
        dn = both[1, :]; gn = both[2, :]
        mask = use_gpu ? CuArray(reshape(active, 1, Nf)) : reshape(active, 1, Nf)
        for c in comps
            g[c] .= ifelse.(mask, gnew[c], g[c])
        end
        done = sqrt.(dn) ./ sqrt.(gn) .< tolerance
        active .&= rho > 0 ? .!fill(all(done[active]), Nf) : .!done     # coupled channels stop together
    end
    !any(active), niters, 1 - count(active) / Nf
end

"""
    square_model(model, meta; minuvw=0.0, precision=:double, wideband=false) -> GPUSquareVisibilities

The square (masked) model of one direction as `solve_gains!` builds it, to pass
as its `model_sq` keyword when the same model is solved for repeatedly.
"""
function square_model(model::GPUVisibilities, meta::GPUMetadata; minuvw::Real=0.0,
                      precision::Symbol=:double, wideband::Bool=false)
    use_gpu = _is_gpu(model)
    T = precision == :mixed && !wideband ? ComplexF32 : ComplexF64
    sq = GPUSquareVisibilities(meta.Nant, wideband ? 1 : Nfreq(model), gpu=use_gpu, T=T)
    if use_gpu
        square_visibilities!(sq, model, meta, minuvw)
    else
        host_makesquare!(sq, model, meta); mask_short_baselines!(sq, meta, minuvw)
    end
    sq
end
