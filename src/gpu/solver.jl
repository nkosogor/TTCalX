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
        a .= ifelse.(keep, a, zero(ComplexF64))
    end
    sq
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
    nxx = ComplexF64(0); nxy = ComplexF64(0); nyx = ComplexF64(0); nyy = ComplexF64(0)
    dxx = ComplexF64(0); dxy = ComplexF64(0); dyx = ComplexF64(0); dyy = ComplexF64(0)
    for i in 1:Nant
        gi_xx = gxx[i, β]; gi_xy = gxy[i, β]; gi_yx = gyx[i, β]; gi_yy = gyy[i, β]
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
    nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy
end

"""
Full-Jones StefCal step for antenna j, channel β: (D⁻¹N)ᴴ - G_j, or zero
when D is singular.  With a prior (`start !== nothing`) each diagonal term
of X = G_jᴴ is regularized on the Schur complement of its column's quadratic
cost, and the off-diagonal term is moved to its conditional optimum.
"""
@inline function _full_step(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                            start, width, lam, Nant)
    nxx, nxy, nyx, nyy, dxx, dxy, dyx, dyy =
        _full_normal_equations(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, Nant)
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

"""Diagonal StefCal step for antenna j, channel β (as `stefcal_step_diagonal_kernel!`), optionally regularized."""
@inline function _diag_step(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
    nxx = ComplexF64(0); nyy = ComplexF64(0); dxx = ComplexF64(0); dyy = ComplexF64(0)
    for i in 1:Nant
        gi_xx = gxx[i, β]; gi_yy = gyy[i, β]
        gm_xx = gi_xx * Mxx[i, j, β]; gm_xy = gi_xx * Mxy[i, j, β]
        gm_yx = gi_yy * Myx[i, j, β]; gm_yy = gi_yy * Myy[i, j, β]
        nxx += conj(gm_xx) * Vxx[i, j, β] + conj(gm_yx) * Vyx[i, j, β]
        nyy += conj(gm_xy) * Vxy[i, j, β] + conj(gm_yy) * Vyy[i, j, β]
        dxx += conj(gm_xx) * gm_xx + conj(gm_yx) * gm_yx
        dyy += conj(gm_xy) * gm_xy + conj(gm_yy) * gm_yy
    end
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
                                   start, width, lam, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        sxx, sxy, syx, syy = _full_step(j, β, gxx, gxy, gyx, gyy, Vxx, Vxy, Vyx, Vyy,
                                        Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
        kxx[j, β] = sxx; kxy[j, β] = sxy; kyx[j, β] = syx; kyy[j, β] = syy
    end
    return nothing
end

function stefcal_diag_step_kernel!(kxx, kyy, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy,
                                   start, width, lam, Nant, Nfreq)
    idx = thread_index_1d()
    if idx <= Nant * Nfreq
        j = ((idx - 1) % Nant) + 1
        β = ((idx - 1) ÷ Nant) + 1
        sxx, syy = _diag_step(j, β, gxx, gyy, Vxx, Vxy, Vyx, Vyy, Mxx, Mxy, Myx, Myy, start, width, lam, Nant)
        kxx[j, β] = sxx; kyy[j, β] = syy
    end
    return nothing
end

"""
    stefcal_steps!(k, g, is_diagonal, meas_sq, model_sq, Nant; reg=nothing)

StefCal steps k = G_new(g) - g for all antennas and channels, for gains
`g = (xx, xy, yx, yy)` (`xy`, `yx` unused when diagonal), on the arrays'
device.  `reg` is an optional `PhaseRegularization`.
"""
function stefcal_steps!(k, g, is_diagonal::Bool, meas_sq::GPUSquareVisibilities,
                        model_sq::GPUSquareVisibilities, Nant::Int; reg=nothing)
    Nf = size(g[1], 2)
    start, width, lam = reg === nothing ? (nothing, nothing, 0.0) : (reg.start, reg.width, reg.lambda)
    V = (meas_sq.xx, meas_sq.xy, meas_sq.yx, meas_sq.yy)
    M = (model_sq.xx, model_sq.xy, model_sq.yx, model_sq.yy)
    if g[1] isa CuArray
        blocks, threads = kernel_config_1d(Nant * Nf)
        if is_diagonal
            @cuda blocks=blocks threads=threads stefcal_diag_step_kernel!(
                k[1], k[4], g[1], g[4], V..., M..., start, width, lam, Nant, Nf)
        else
            @cuda blocks=blocks threads=threads stefcal_full_step_kernel!(
                k..., g..., V..., M..., start, width, lam, Nant, Nf)
        end
        CUDA.synchronize()
    else
        Threads.@threads for β in 1:Nf
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
#                                 Solver                                       #
#==============================================================================#

_channel_sq(a) = vec(Array(sum(abs2.(a), dims=1)))

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
them from both sides).  `reg` is an optional `PhaseRegularization`.  Works on
GPU arrays and, per channel only, on host arrays (one channel per thread).
`converged` is true when every channel converged.
"""
function solve_gains!(calibration::GPUCalibration, measured::GPUVisibilities,
                      model::GPUVisibilities, meta::GPUMetadata;
                      solver::Symbol=:rk4, maxiter::Integer=5, tolerance::Real=1e-4,
                      minuvw::Real=0.0, reg=nothing)
    solver in (:rk4, :damped) || throw(ArgumentError("unknown solver $solver (use :rk4 or :damped)"))
    tolerance, minuvw = Float64(tolerance), Float64(minuvw)
    use_gpu = _is_gpu(measured)
    Na = meta.Nant
    wideband = Nfreq(calibration) == 1 && Nfreq(measured) > 1
    !use_gpu && wideband && throw(ArgumentError("host solve_gains! supports per-channel calibrations only"))
    if reg !== nothing
        size(reg.start) == (Na, Nfreq(calibration)) || throw(ArgumentError(
            "phase prior is $(size(reg.start)), calibration is $((Na, Nfreq(calibration))) " *
            "(use phase_regularization(...; nfreq=1) for wideband calibrations)"))
        (reg.start isa CuArray) == _is_gpu(calibration) || throw(ArgumentError(
            "phase prior and calibration must be on the same device (phase_regularization(...; gpu=...))"))
    end
    meas_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(measured), gpu=use_gpu)
    model_sq = GPUSquareVisibilities(Na, wideband ? 1 : Nfreq(model), gpu=use_gpu)
    makesquare! = use_gpu ? gpu_makesquare! : host_makesquare!
    makesquare!(meas_sq, measured, meta)
    makesquare!(model_sq, model, meta)
    mask_short_baselines!(meas_sq, meta, minuvw)
    mask_short_baselines!(model_sq, meta, minuvw)

    Nf = Nfreq(calibration)
    zero_like() = ntuple(_ -> use_gpu ? CUDA.zeros(ComplexF64, Na, Nf) : zeros(ComplexF64, Na, Nf), 4)
    g = (calibration.xx, calibration.xy, calibration.yx, calibration.yy)
    diag = calibration.is_diagonal
    comps = diag ? (1, 4) : (1, 2, 3, 4)
    steps!(k, x) = stefcal_steps!(k, x, diag, meas_sq, model_sq, Na; reg=reg)

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
        dn = zeros(Nf); gn = zeros(Nf)
        for c in comps
            dn .+= _channel_sq(gnew[c] .- g[c])
            gn .+= _channel_sq(g[c])
        end
        mask = use_gpu ? CuArray(reshape(active, 1, Nf)) : reshape(active, 1, Nf)
        for c in comps
            g[c] .= ifelse.(mask, gnew[c], g[c])
        end
        active .&= .!(sqrt.(dn) ./ sqrt.(gn) .< tolerance)
    end
    !any(active), niters, 1 - count(active) / Nf
end
