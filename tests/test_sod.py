"""Sod shock tube along each axis vs the exact Riemann solution; the three axes must agree to roundoff."""
import numpy as np
from common import run, field, last_step, centers, check
from riemann import exact_rho

n, tol = 200, float(__import__("os").environ.get("SOD_TOL", "1e-2"))
rho = {}
for ax in range(3):
    N = [4, 4, 4]; N[ax] = n
    d, r = run(f"sod{ax}", 1, case="sod", axis=ax, nx=N[0], ny=N[1], nz=N[2], nout=10**6, ndiag=10**6)
    q = field(r, last_step(r), N)
    rho[ax] = np.moveaxis(q[0], 2 - ax, 0)[:, 0, 0]          # line along the shock-tube axis
    check(np.isfinite(q).all(), f"axis {ax}: finite fields")
    check(abs(d[-1, 1] - 0.2) < 1e-12, f"axis {ax}: reached t=0.2")
ex = exact_rho(centers(0, 1, n), 0.2)
err = np.abs(rho[0] - ex).mean()
print(f"Sod L1 density error at n={n}: {err:.4e}")
check(err < tol, f"L1 error {err:.3e} < {tol}")
check(np.abs(rho[1] - rho[0]).max() < 1e-12 and np.abs(rho[2] - rho[0]).max() < 1e-12, "x, y, z axes agree")
print("PASS test_sod")
