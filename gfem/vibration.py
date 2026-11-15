"""Spectrum-based dynamics from modal results (all vectorised, device friendly).

* response_spectrum : single-point response spectrum (seismic / shock), SRSS or CQC
                      combination, mode participation in any direction  (ANSYS RS, Nastran)
* random_vibration  : base-acceleration PSD -> 1-sigma RMS displacement and stress (modal
                      covariance on the device: cost ~ nfreq*nmodes^2 + nmodes^2*ndof)
"""
import numpy as np
import jax.numpy as jnp

from . import solvers as S


def _modal(model, step, nmodes, state=None, monitor=None):
    st = S.Step(kind="frequency", bcs=step.bcs, nmodes=nmodes, eigensolver=getattr(step, "eigensolver", "auto"))
    return S.modal(model, st, state, monitor)


def _mode_stresses(model, modes, states):
    from .analysis import postprocess
    out = []
    for phi in modes:
        fr = postprocess(model, jnp.asarray(phi), states)
        out.append(fr["S"])
    return np.stack(out)                                        # (nm, nn, 6)


def _von_mises(Sv):
    s = Sv
    return np.sqrt(np.maximum(0.5 * ((s[..., 0] - s[..., 1]) ** 2 + (s[..., 1] - s[..., 2]) ** 2 + (s[..., 2] - s[..., 0]) ** 2)
                              + 3 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2), 0))


def cqc_rho(w, zeta):
    r = w[None, :] / w[:, None]
    return 8 * zeta ** 2 * (1 + r) * r ** 1.5 / ((1 - r ** 2) ** 2 + 4 * zeta ** 2 * r * (1 + r) ** 2)


def response_spectrum(model, step, spectrum, direction=(0, 0, 1), nmodes=20, damping=0.05,
                      combination="SRSS", res=None, monitor=None):
    """spectrum: (n, 2) table of (frequency Hz, spectral acceleration). Returns dict with
    combined displacement U (nn, 3), stress S (nn, 6), MISES, per-mode data."""
    res = res or _modal(model, step, nmodes, monitor=monitor)
    f = np.asarray(res["freq"])
    w = 2 * np.pi * np.maximum(f, 1e-12)
    spec = np.asarray(spectrum, float).reshape(-1, 2)
    Sa = np.interp(f, spec[:, 0], spec[:, 1])
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    gam = res["participation"] @ d
    q = gam * Sa / w ** 2                                       # modal amplitudes
    modes = np.asarray(res["modes"])
    nr = getattr(model, "n_real", model.nn)
    Um = (q[:, None] * modes).reshape(len(f), -1, 3)[:, :nr]
    Sm = q[:, None, None] * _mode_stresses(model, modes, model.init_state())
    if combination.upper() == "CQC":
        rho = cqc_rho(w, damping)
        comb = lambda A: np.sqrt(np.maximum(np.einsum("i...,ij,j...->...", A, rho, A), 0))
    else:
        comb = lambda A: np.sqrt(np.sum(A * A, axis=0))
    U = comb(Um)
    Sc = comb(Sm)
    return dict(U=U, S=Sc, MISES=_von_mises(Sc), freq=f, participation=gam, modal=res,
                Umode=Um, combination=combination)


def random_vibration(model, step, psd, direction=(0, 0, 1), nmodes=20, damping=0.02, nfreq=400,
                     res=None, monitor=None):
    """psd: (n, 2) base-acceleration PSD table (frequency Hz, accel^2/Hz), log-log interpolated.
    Returns 1-sigma RMS relative displacement (nn, 3), RMS stress components and an equivalent
    RMS von Mises (Segalman-style from component RMS)."""
    res = res or _modal(model, step, nmodes, monitor=monitor)
    f = np.asarray(res["freq"])
    w = 2 * np.pi * np.maximum(f, 1e-12)
    tab = np.asarray(psd, float).reshape(-1, 2)
    fmin, fmax = tab[0, 0], tab[-1, 0]
    grid = np.geomspace(fmin, fmax, nfreq)
    for fj in f[(f > fmin) & (f < fmax)]:                       # resolve every resonance peak
        bw = 2 * damping * fj
        grid = np.concatenate([grid, fj + bw * np.linspace(-4, 4, 41)])
    grid = np.unique(grid[(grid >= fmin) & (grid <= fmax)])
    G = np.exp(np.interp(np.log(grid), np.log(tab[:, 0]), np.log(np.maximum(tab[:, 1], 1e-300))))
    df = np.gradient(grid)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    gam = res["participation"] @ d
    om = 2 * np.pi * grid
    H = -gam[None, :] / (w[None, :] ** 2 - om[:, None] ** 2 + 2j * damping * w[None, :] * om[:, None])
    Hj = jnp.asarray(H)
    C = jnp.real(jnp.einsum("f,fj,fk->jk", jnp.asarray(G * df), Hj, jnp.conj(Hj)))   # modal covariance
    modes = jnp.asarray(res["modes"])
    nr = getattr(model, "n_real", model.nn)
    var_u = jnp.einsum("jd,jk,kd->d", modes, C, modes)
    U = np.sqrt(np.maximum(np.asarray(var_u), 0)).reshape(-1, 3)[:nr]
    Sm = jnp.asarray(_mode_stresses(model, np.asarray(res["modes"]), model.init_state()))
    var_s = jnp.einsum("jnc,jk,knc->nc", Sm, C, Sm)
    Srms = np.sqrt(np.maximum(np.asarray(var_s), 0))
    grms = float(np.sqrt(np.sum(G * df)))
    return dict(U=U, S=Srms, MISES=_von_mises(Srms), freq=f, grms=grms, participation=gam, modal=res,
                grid=grid)
