"""Image-plane deconvolution on the GPU, emulating WSClean/radler for our settings.

* ``SnapshotClean`` + ``clean_snapshot``: Hogbom/Clark CLEAN, as
  ``wsclean -niter 50000 -gain 0.1 -mgain 0.9999 -auto-mask 5 -auto-threshold 1 -local-rms
  -horizon-mask 10deg`` with exactly 2 major cycles (what -mgain 0.9999 gives in practice).
* ``MultiScaleClean`` + ``deconvolve_multiscale``: radler-like multiscale CLEAN with the
  WSClean major-cycle logic for ``-multiscale -multiscale-scale-bias 0.8 -mgain 0.95
  -auto-mask 3 -auto-threshold 0.5 -local-rms -horizon-mask 10deg``.

Images are cupy float32 arrays in WSClean orientation [y, x]; the PSF is N x N with its peak at
(N/2, N/2).  The minor loops run as one CUDA block doing exact Hogbom iterations on a candidate
pixel set (Clark-style: pixels above max(threshold, sidelobe_fraction * peak)); after every
minor loop the full residual is updated exactly by FFT convolution.

Auto-masking (radler): phase 1 cleans to auto_mask * sigma, the pixels (per scale) that received
components become the mask, phase 2 cleans inside the mask to auto_threshold * sigma.
Local RMS (radler RmsImage::Make): sigma map = sqrt(Gaussian-smoothed residual^2), Gaussian FWHM
= 25 theoretical beams; peaks are compared after scaling by min(rms) / rms(x, y).
"""
import math

import numpy as np
import cupy as cp

from .beam import fit_beam

_HOGBOM_SRC = r"""
extern "C" __global__ void hogbom_set(const float* __restrict__ psf, const int pn, const int pc,
    const int* __restrict__ px, const int* __restrict__ py, float* val, const float* __restrict__ fac,
    float* comp, const int K, const float gain, const float thresh, const int maxiter,
    int* out_niter, float* out_peak)
{
    // Hogbom CLEAN restricted to K candidate pixels (px, py); val = residual at those pixels,
    // fac = local-RMS weighting of the peak search; PSF pn x pn with centre pc.
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    int it = 0;
    float peak = 0.f;
    for (; it < maxiter; ++it) {
        float best = -1.f; int bi = -1;
        for (int k = tid; k < K; k += nt) {
            const float a = fabsf(val[k]) * fac[k];
            if (a > best) { best = a; bi = k; }
        }
        sv[tid] = best; si[tid] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            __syncthreads();
        }
        peak = sv[0];
        const int ip = si[0];
        if (peak < thresh || ip < 0) break;
        const float amp = gain * val[ip];
        const int x0 = px[ip], y0 = py[ip];
        __syncthreads();
        if (tid == 0) comp[ip] += amp;
        for (int k = tid; k < K; k += nt) {
            const int dx = px[k] - x0 + pc, dy = py[k] - y0 + pc;
            if (dx >= 0 && dx < pn && dy >= 0 && dy < pn) val[k] -= amp * psf[(size_t)dy * pn + dx];
        }
        __syncthreads();
    }
    if (tid == 0) { *out_niter = it; *out_peak = peak; }
}
"""
_hogbom = cp.RawKernel(_HOGBOM_SRC, "hogbom_set")


def hogbom_on_set(psf, ys, xs, vals, fac, gain, thresh, maxiter):
    """Run the Hogbom kernel; returns (component amplitudes per candidate, n_iter, last peak)."""
    K = int(ys.size)
    comp = cp.zeros(K, cp.float32)
    nit = cp.zeros(1, cp.int32); pk = cp.zeros(1, cp.float32)
    pn = psf.shape[0]
    _hogbom((1,), (1024,), (psf, np.int32(pn), np.int32(pn // 2), xs.astype(cp.int32), ys.astype(cp.int32),
                            vals, fac, comp, np.int32(K), np.float32(gain), np.float32(thresh), np.int32(maxiter), nit, pk))
    return comp, int(nit.get()[0]), float(pk.get()[0])


# ----------------------------------------------------------------------------- helpers
def fast_len(n):
    """smallest 2^a 3^b 5^c 7^d >= n"""
    best = 1 << int(math.ceil(math.log2(n)))
    for a in (1, 3, 9, 27, 81, 243):
        for b in (1, 5, 25, 125, 625, 3125):
            for c in (1, 7, 49, 343):
                m = a * b * c
                if m > best:
                    continue
                p = m
                while p < n:
                    p *= 2
                best = min(best, p)
    return best


class Convolver:
    """Linear (non-cyclic) convolution of N x N images with M x M kernels centred at M//2,
    by zero-padded real FFTs of size P = fast_len(N + M//2 + extra)."""

    def __init__(self, N, M=None, extra=2):
        self.N = N
        self.M = N if M is None else M
        self.P = fast_len(N + self.M // 2 + extra)

    def kfft(self, k):
        P, M = self.P, k.shape[0]; c = M // 2
        kp = cp.zeros((P, P), cp.float32)
        k = cp.asarray(k, dtype=cp.float32)
        kp[:M - c, :M - c] = k[c:, c:]; kp[:M - c, P - c:] = k[c:, :c]
        kp[P - c:, :M - c] = k[:c, c:]; kp[P - c:, P - c:] = k[:c, :c]
        return cp.fft.rfft2(kp)

    def img_fft(self, img):
        x = cp.zeros((self.P, self.P), cp.float32); x[:self.N, :self.N] = img
        return cp.fft.rfft2(x)

    def back(self, F):
        return cp.fft.irfft2(F, s=(self.P, self.P))[:self.N, :self.N]

    def __call__(self, img, K):
        return self.back(self.img_fft(img) * K)


def gaussian_kernel(N, bmaj_deg, bmin_deg, bpa_deg, scale_deg):
    """Peak-1 elliptical Gaussian (2h+1)^2 in [y, x] orientation; BPA from +y (north) towards -x
    (east for CDELT1 < 0), as BPA in the FITS header."""
    s2f = 2.0 * math.sqrt(2.0 * math.log(2.0))
    smaj = bmaj_deg / scale_deg / s2f
    smin = bmin_deg / scale_deg / s2f
    half = int(min(N // 2 - 1, math.ceil(5 * smaj)))
    y, x = cp.mgrid[-half:half + 1, -half:half + 1].astype(cp.float32)
    pa = math.radians(bpa_deg)
    a = -x * math.sin(pa) + y * math.cos(pa)   # along the major axis, unit vector (-sin pa, cos pa)
    b = x * math.cos(pa) + y * math.sin(pa)    # along the minor axis
    k = cp.exp(-0.5 * (a * a / smaj ** 2 + b * b / smin ** 2)).astype(cp.float32)
    return k, half


def horizon_mask(N, scale_deg, horizon_deg=10.0):
    """True where elevation > horizon_deg (r = sqrt(l^2 + m^2) <= cos(horizon))."""
    j = cp.arange(N, dtype=cp.float32) - N // 2
    s = math.radians(scale_deg)
    r2 = (j[None, :] * s) ** 2 + (j[:, None] * s) ** 2
    return r2 <= math.cos(math.radians(horizon_deg)) ** 2


def rms_image(img, beam_fwhm_pix, window=25.0):
    """radler RmsImage::Make: sqrt(Gaussian-weighted mean of img^2), Gaussian FWHM = window*beam
    (cyclic FFT smoothing with a unit-sum Gaussian)."""
    N = img.shape[0]
    sig = window * beam_fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    f = cp.fft.fftfreq(N).astype(cp.float32)
    g = cp.exp(-2.0 * (math.pi * sig) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2)).astype(cp.float32)
    sm = cp.fft.ifft2(cp.fft.fft2(img.astype(cp.float32) ** 2) * g).real
    return cp.sqrt(cp.maximum(sm, 0)).astype(cp.float32)


def local_rms_factor(res, theo_pix, scale_deg, horizon, window=25.0):
    """(fac, min_rms): fac = min_rms / rms(x, y).  The minimum is taken over the horizon mask,
    excluding pixels whose RMS window reaches the zeroed region beyond the horizon (r >= 1)."""
    N = res.shape[0]
    rms = rms_image(res, theo_pix, window)
    jj = cp.arange(N, dtype=cp.float32) - N // 2
    far = cp.sqrt(jj[None, :] ** 2 + jj[:, None] ** 2) < (1.0 / math.radians(scale_deg) - 3 * window * theo_pix / 2.3548)
    minrms = float(rms[horizon & far].min())
    return (minrms / cp.maximum(rms, 1e-30)).astype(cp.float32), minrms


def mad_std(x):
    """radler StdDevFromMAD (median of |x|, i.e. about zero)."""
    return float(1.4826 * cp.median(cp.abs(x)))


# ----------------------------------------------------------------------------- snapshot (Clark)
class SnapshotClean:
    def __init__(self, psf, N, scale_deg, theo_pix, beam=None, gain=0.1, horizon_deg=10.0, kmax=60000,
                 log=None):
        self.N = N; self.scale = scale_deg
        self.psf = cp.asarray(psf, dtype=cp.float32)
        self.cv = Convolver(N, self.psf.shape[0], extra=1)
        self.Kpsf = self.cv.kfft(self.psf)
        self.gain = gain
        self.mask = horizon_mask(N, scale_deg, horizon_deg)
        self.beam = tuple(beam) if beam is not None else fit_beam(self.psf, scale_deg, theo_pix)
        self.kmax = kmax
        self.log = log or (lambda *a: None)
        # Clark candidate fraction: peak |PSF| outside ~2.5 restoring-beam FWHM of the centre
        p = self.psf; c = p.shape[0] // 2
        hw = max(3, int(2.5 * self.beam[0] / scale_deg))
        q = cp.abs(p).copy(); q[c - hw:c + hw + 1, c - hw:c + hw + 1] = 0
        self.clark_frac = float(q.max())

    def conv(self, img):
        return self.cv(img, self.Kpsf)

    def clean_full(self, res, model, thresh, fac=None, maxiter=50000, niter_done=0):
        """Clark cycles over the whole (horizon-masked) image until the scaled peak < thresh."""
        N = self.N
        facimg = cp.ones((N, N), cp.float32) if fac is None else fac
        nit = niter_done; ncyc = 0
        while nit < maxiter:
            a = cp.where(self.mask, cp.abs(res) * facimg, 0)
            peak = float(a.max())
            if peak < thresh:
                break
            t = max(thresh, peak * self.clark_frac)
            sel = a >= t
            if int(sel.sum()) > self.kmax:
                t = float(cp.sort(a[sel])[-self.kmax]); sel = a >= t
            ys, xs = cp.nonzero(sel)
            comp, n, _ = hogbom_on_set(self.psf, ys, xs, res[ys, xs].astype(cp.float32),
                                       facimg[ys, xs].astype(cp.float32), self.gain, t, maxiter - nit)
            nit += n; ncyc += 1
            dm = cp.zeros((N, N), cp.float32); dm[ys, xs] = comp
            model += dm
            res = res - self.conv(dm)
            if n == 0:
                break
        self.log(f"  clean_full: {nit - niter_done} iters, {ncyc} Clark cycles, thresh {thresh:.4g}, frac {self.clark_frac:.3f}")
        return model, res, nit

    def clean_masked(self, res, model, thresh, maskpix, maxiter=50000, niter_done=0):
        """Hogbom restricted to a fixed pixel set (auto-mask phase 2), exact, one kernel launch."""
        ys, xs = cp.nonzero(maskpix & self.mask)
        comp, n, _ = hogbom_on_set(self.psf, ys, xs, res[ys, xs].astype(cp.float32), cp.ones(ys.size, cp.float32),
                                   self.gain, thresh, maxiter - niter_done)
        dm = cp.zeros_like(model); dm[ys, xs] = comp
        model += dm
        res = res - self.conv(dm)
        self.log(f"  clean_masked: {n} iters on {ys.size} pixels, thresh {thresh:.4g}")
        return model, res, niter_done + n

    def restore(self, model, residual):
        k, h = gaussian_kernel(self.N, *self.beam, self.scale)
        cv = Convolver(self.N, 2 * h + 1, extra=1)
        return residual + cv(model, cv.kfft(k))


def clean_snapshot(dirty, psf, residual_fn, scale_deg, theo_pix, sky, params, beam=None, log=None):
    """Snapshot CLEAN driver (radler auto-mask logic with a fixed number of major cycles).

    phase 1: Clark clean to auto_mask * min(local rms) (local rms of the dirty image)
    major 1: exact residual from the visibilities
    phase 2: Hogbom inside the phase-1 component mask to auto_threshold * MAD-sigma
    major 2: exact residual; restore with the fitted beam.
    residual_fn(model) -> residual image.  sky: pixels inside the horizon (r < 1) for the MAD.
    Returns (restored, residual, model, info)."""
    N = dirty.shape[0]
    D = SnapshotClean(psf, N, scale_deg, theo_pix, beam=beam, gain=params.gain, horizon_deg=params.horizon_deg, log=log)
    model = cp.zeros((N, N), cp.float32)
    res = dirty.copy()
    sig0 = mad_std(res[sky])
    fac, minrms = local_rms_factor(res, theo_pix, scale_deg, D.mask, params.local_rms_window)
    t1 = params.auto_mask * minrms
    model, res, nit = D.clean_full(res, model, t1, fac=fac, maxiter=params.niter)
    nmaj = 0
    if params.nmajor >= 1:
        res = residual_fn(model); nmaj += 1
    sig1 = mad_std(res[sky])
    t2 = params.auto_threshold * sig1
    model, res, nit = D.clean_masked(res, model, t2, model != 0, maxiter=params.niter, niter_done=nit)
    if params.nmajor >= 2:
        res = residual_fn(model); nmaj += 1
    restored = D.restore(model, res)
    info = dict(niter=nit, nmajor=nmaj, thr1=t1, thr2=t2, minrms=minrms, sig0=sig0, sig1=sig1, beam=D.beam,
                ncomp=int(cp.count_nonzero(model)), clark_frac=D.clark_frac)
    return restored, res, model, info


# ----------------------------------------------------------------------------- multiscale (hour)
def shape_kernel(N, scale):
    """radler tapered-quadratic scale kernel of width `scale` px (unit sum), N x N centred."""
    c = N // 2
    k = cp.zeros((N, N), cp.float32)
    if scale == 0:
        k[c, c] = 1.0; return k
    R = 0.5 * scale; h = int(math.ceil(R))
    y, x = cp.mgrid[-h:h + 1, -h:h + 1].astype(cp.float32)
    rn = cp.sqrt(x * x + y * y) / R
    f = cp.where(rn < 1, (1 - rn * rn) * 0.5 * (1 + cp.cos(cp.pi * rn)), 0).astype(cp.float32)
    k[c - h:c + h + 1, c - h:c + h + 1] = f / f.sum()
    return k


class MultiScaleClean:
    """Scales 0 and 4 x theoretical beam, doubling up to N/2 (radler defaults).  Scale choice:
    argmax_s max|S_s * R| * fac * bias^-s (bias = scale_bias^-index, favours large scales);
    sub-minor loop = Hogbom on the candidate set of that scale with the twice-convolved PSF
    S_s * S_s * PSF, then exact FFT update of the full residual with S_s * PSF."""

    def __init__(self, psf, N, scale_deg, theo_pix, beam=None, scale_bias=0.8, gain=0.1, horizon_deg=10.0,
                 kmax=60000, max_scale_frac=0.5, multiscale=True, log=None):
        self.N = N; self.sdeg = scale_deg; self.gain = gain; self.kmax = kmax
        self.log = log or (lambda *a: None)
        self.psf = cp.asarray(psf, dtype=cp.float32)
        self.beam = tuple(beam) if beam is not None else fit_beam(self.psf, scale_deg, theo_pix)
        self.mask = horizon_mask(N, scale_deg, horizon_deg)
        b = self.beam[0] / scale_deg
        scales = [0]
        if multiscale:
            s = 4 * theo_pix
            while s < N * max_scale_frac:
                scales.append(s); s *= 2
        self.scales = scales
        self.C = Convolver(N, N, extra=2)
        Kpsf = self.C.kfft(self.psf)
        self.KS, self.KP, self.Pss, self.pss0, self.bias = [], [], [], [], []
        c = N // 2
        for k, s in enumerate(scales):
            KS = self.C.kfft(shape_kernel(N, s))
            KP = KS * Kpsf
            delta = cp.zeros((N, N), cp.float32); delta[c, c] = 1
            Ps_img = self.C(delta, KP)
            Pss_img = self.C(Ps_img, KS)
            self.KS.append(KS); self.KP.append(KP)
            self.pss0.append(float(Pss_img[c, c]))
            self.Pss.append((Pss_img / float(Pss_img[c, c])).astype(cp.float32))
            self.bias.append((1.0 / scale_bias) ** k)
        self.frac = []          # Clark fraction per scale: peak sidelobe of the normalised P_ss
        for k, s in enumerate(scales):
            q = cp.abs(self.Pss[k]).copy()
            hw = max(3, int(2.5 * b + s))
            q[c - hw:c + hw + 1, c - hw:c + hw + 1] = 0
            self.frac.append(float(q.max()))
        self.log(f"MultiScaleClean: beam {self.beam[0]*60:.2f}'x{self.beam[1]*60:.2f}' ({b:.1f} px), "
                 f"scales {[round(s, 1) for s in scales]}, clark frac {[round(f, 3) for f in self.frac]}")

    def scale_images(self, R):
        F = self.C.img_fft(R)
        return [R if s == 0 else self.C.back(F * self.KS[k]) for k, s in enumerate(self.scales)]

    def minor(self, R, M, stop, fac, scale_masks=None, maxiter=500000, niter_done=0, comp_masks=None,
              max_cycles=100000):
        """Clean until all scaled peaks < stop.  scale_masks: per-scale allowed pixels (phase 2)."""
        N = self.N; nit = niter_done; ncyc = 0
        if comp_masks is None:
            comp_masks = [cp.zeros((N, N), bool) for _ in self.scales]
        allow = [self.mask if scale_masks is None else (self.mask & scale_masks[k]) for k in range(len(self.scales))]
        per_scale = [0] * len(self.scales)
        while nit < maxiter and ncyc < max_cycles:
            Rs = self.scale_images(R)
            absf = [cp.where(allow[k], cp.abs(Rs[k]) * fac, 0) for k in range(len(self.scales))]
            pk = [float(a.max()) for a in absf]
            score = [pk[k] * self.bias[k] for k in range(len(self.scales))]
            k = int(np.argmax(score))
            if score[k] < stop:
                break
            t = max(stop / self.bias[k], pk[k] * self.frac[k])
            sel = absf[k] >= t
            if int(sel.sum()) > self.kmax:
                t = float(cp.sort(absf[k][sel])[-self.kmax]); sel = absf[k] >= t
            ys, xs = cp.nonzero(sel)
            vals = (Rs[k][ys, xs] / self.pss0[k]).astype(cp.float32)
            comp, n, _ = hogbom_on_set(self.Pss[k], ys, xs, vals, fac[ys, xs].astype(cp.float32), self.gain,
                                       t / self.pss0[k], maxiter - nit)
            if n == 0:
                break
            nit += n; ncyc += 1; per_scale[k] += n
            dm = cp.zeros((N, N), cp.float32); dm[ys, xs] = comp
            comp_masks[k] |= dm != 0
            Fd = self.C.img_fft(dm)
            M += dm if self.scales[k] == 0 else self.C.back(Fd * self.KS[k])
            R = R - self.C.back(Fd * self.KP[k])
        self.log(f"   minor: {nit - niter_done} iters, {ncyc} cycles, stop {stop:.4g}, per scale {per_scale}")
        return R, M, nit, comp_masks

    def restore(self, M, R):
        k, h = gaussian_kernel(self.N, *self.beam, self.sdeg)
        big = cp.zeros((self.N, self.N), cp.float32); c = self.N // 2
        big[c - h:c + h + 1, c - h:c + h + 1] = k
        return R + self.C(M, self.C.kfft(big))


def deconvolve_multiscale(dirty, psf, scale_deg, theo_pix, params, residual_fn=None, beam=None, max_major=20,
                          log=None):
    """WSClean-like major-cycle driver for multiscale CLEAN with auto-mask/auto-threshold.

    Each minor loop stops at max(threshold, (1 - mgain) * peak); the residual is then recomputed
    from the visibilities by residual_fn(model).  When a minor loop reaches the auto-mask
    threshold, the component positions become the mask and the threshold drops to
    auto_threshold * sigma (phase 2); stops when the peak inside the mask is below it.
    Returns (restored, residual, model, info)."""
    log = log or (lambda *a: None)
    N = dirty.shape[0]
    D = MultiScaleClean(psf, N, scale_deg, theo_pix, beam=beam, scale_bias=params.multiscale_scale_bias,
                        gain=params.gain, horizon_deg=params.horizon_deg, multiscale=params.multiscale, log=log)
    sky = D.mask
    # pixels above the horizon (r < 1), the region of the MAD noise estimate (as clean_snapshot)
    above = horizon_mask(N, scale_deg, 0.0)
    R = dirty.copy(); M = cp.zeros((N, N), cp.float32)
    # auto_mask = 0: no auto-mask phase (phase 2 from the start, unmasked); the stopping threshold
    # is max(threshold, auto_threshold * sigma) as WSClean's -threshold / -auto-threshold
    nit = 0; phase = 1 if params.auto_mask else 2; masks = None; comp_masks = None; nmajor = 0
    info = dict(majors=[])
    while True:
        # radler applies -local-rms only while auto-masking (phase 1): its logs print "Lowest RMS
        # in image" for the first deconvolution only, and in the auto-threshold phase (phase 2)
        # thresholds use the plain MAD noise ("Estimated standard deviation") with unweighted peaks.
        if params.local_rms and phase == 1:
            fac, sigma = local_rms_factor(R, theo_pix, scale_deg, sky, params.local_rms_window)
        else:
            fac = cp.ones((N, N), cp.float32); sigma = mad_std(R[above])
        region = sky if (phase == 1 or not masks) else (sky & cp.stack(masks).any(axis=0))
        peak = float(cp.where(region, cp.abs(R) * fac, 0).max())
        thr = max(params.threshold, (params.auto_mask if phase == 1 else params.auto_threshold) * sigma)
        stop = max(thr, (1 - params.mgain) * peak) if residual_fn is not None else thr
        log(f" major {nmajor}: peak {peak:.4g} sigma {sigma:.4g} phase {phase} stop {stop:.4g}")
        info["majors"].append(dict(peak=peak, sigma=sigma, phase=phase, stop=stop, nit=nit))
        if peak < thr and phase == 2:
            break
        if peak < thr and phase == 1:
            phase = 2; masks = [m.copy() for m in comp_masks] if comp_masks else None
            continue
        final = stop <= thr * 1.0000001
        R, M, nit, comp_masks = D.minor(R, M, stop, fac, scale_masks=masks if phase == 2 else None,
                                        maxiter=params.niter, niter_done=nit, comp_masks=comp_masks)
        did = nit - info["majors"][-1]["nit"]
        if residual_fn is not None and did > 0:
            R = residual_fn(M); nmajor += 1
        if did == 0 and phase == 2:
            break
        if nit >= params.niter or nmajor >= max_major:
            break
        if final:     # this minor loop went down to the phase threshold (WSClean "(final)")
            if phase == 1:
                phase = 2; masks = [m.copy() for m in comp_masks]
                continue
            break
    info.update(nit=nit, nmajor=nmajor, beam=D.beam, scales=D.scales, niter_reached=nit >= params.niter,
                max_major_reached=nmajor >= max_major)
    return D.restore(M, R), R, M, info
