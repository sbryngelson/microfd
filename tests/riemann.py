"""Exact Riemann solution density for an ideal gas (Toro, ch. 4). Used by the Sod test."""
import numpy as np

def exact_rho(x, t, g=1.4, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), x0=0.5):
    rl, ul, pl = left; rr, ur, pr = right
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def fk(p, rk, pk, ck):                       # pressure function and its derivative for one side
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B)), np.sqrt(A / (p + B)) * (1 - (p - pk) / (2 * (p + B)))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1), (p / pk) ** (-(g + 1) / (2 * g)) / (rk * ck)

    p = 0.5 * (pl + pr)
    for _ in range(100):                         # Newton iteration for the star pressure
        fl, dl = fk(p, rl, pl, cl); fr, dr = fk(p, rr, pr, cr)
        p = max(p - (fl + fr + ur - ul) / (dl + dr), 1e-12)
    u = 0.5 * (ul + ur) + 0.5 * (fk(p, rr, pr, cr)[0] - fk(p, rl, pl, cl)[0])

    s = (np.asarray(x) - x0) / t
    rho = np.empty_like(s)
    for side in (-1, 1):                         # -1: left of the contact, +1: right of it
        rk, uk, pk, ck = (rl, ul, pl, cl) if side < 0 else (rr, ur, pr, cr)
        m = s <= u if side < 0 else s > u
        sm = s[m]
        if p > pk:                               # shock
            rs = rk * (p / pk + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pk + 1)
            S = uk + side * ck * np.sqrt((g + 1) / (2 * g) * p / pk + (g - 1) / (2 * g))
            rho[m] = np.where(side * (sm - S) > 0, rk, rs)
        else:                                    # rarefaction
            rs, cs = rk * (p / pk) ** (1 / g), ck * (p / pk) ** ((g - 1) / (2 * g))
            Sh, St = uk + side * ck, u + side * cs
            fan = rk * (2 / (g + 1) - side * (g - 1) / ((g + 1) * ck) * (uk - sm)) ** (2 / (g - 1))
            rho[m] = np.where(side * (sm - Sh) > 0, rk, np.where(side * (sm - St) < 0, rs, fan))
    return rho
