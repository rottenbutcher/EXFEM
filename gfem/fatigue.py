"""Stress-life (S-N) fatigue post-processing (ANSYS Fatigue Tool / nCode-like basics).

Equivalent stress: signed von Mises (sign of the largest-magnitude principal stress).
Loading: fully reversed, zero-based or a constant ratio R = s_min / s_max, scaled by `scale`.
Mean-stress correction: none, Goodman, Gerber, Soderberg (reduce to an equivalent fully reversed
amplitude). Basquin S-N curve  s_ar = Sf (2N)^b  with an endurance limit Se (infinite life).
"""
import numpy as np


def signed_von_mises(Sv):
    from .analysis import principal
    P = principal(Sv)
    vm = np.sqrt(np.maximum(0.5 * ((Sv[:, 0] - Sv[:, 1]) ** 2 + (Sv[:, 1] - Sv[:, 2]) ** 2 + (Sv[:, 2] - Sv[:, 0]) ** 2)
                            + 3 * (Sv[:, 3] ** 2 + Sv[:, 4] ** 2 + Sv[:, 5] ** 2), 0))
    big = np.where(np.abs(P[:, 0]) >= np.abs(P[:, 2]), P[:, 0], P[:, 2])
    return np.where(big >= 0, vm, -vm)


def stress_life(Sv, Su, Sf, b, Se=0.0, Sy=None, loading="fully_reversed", R=-1.0, scale=1.0,
                mean="goodman", design_life=1e7, max_life=1e9):
    s = signed_von_mises(np.asarray(Sv)) * scale
    if loading == "fully_reversed":
        R = -1.0
    elif loading == "zero_based":
        R = 0.0
    smax = np.where(s >= 0, s, -s)
    sa = 0.5 * (1 - R) * smax
    sm = 0.5 * (1 + R) * s
    m = mean.lower()
    with np.errstate(divide="ignore", invalid="ignore"):
        if m == "goodman":
            sar = sa / np.clip(1 - np.maximum(sm, 0) / Su, 1e-9, None)
        elif m == "gerber":
            sar = sa / np.clip(1 - (sm / Su) ** 2, 1e-9, None)
        elif m == "soderberg":
            sar = sa / np.clip(1 - np.maximum(sm, 0) / (Sy or Su), 1e-9, None)
        else:
            sar = sa
        life = 0.5 * (sar / Sf) ** (1.0 / b)
    life = np.where(sar <= max(Se, 1e-300), max_life, np.minimum(life, max_life))
    life = np.where(sar <= 0, max_life, life)
    s_design = max(Sf * (2 * design_life) ** b, Se)
    sf = np.minimum(s_design / np.maximum(sar, 1e-300), 15.0)
    return dict(life=life, damage=design_life / life, safety_factor=sf, alternating=sar)
