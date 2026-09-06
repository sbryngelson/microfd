"""Reflecting walls: Sedov blast in the positive octant with walls must match the octant of the full-domain run."""
import numpy as np
from common import run, field, last_step, check

n = 32
df, rf = run("sedov_full", 1, case="sedov", nx=2*n, ny=2*n, nz=2*n, tend=0.05, nout=10**6, ndiag=10**6)
do, ro = run("sedov_oct", 1, case="sedov", nx=n, ny=n, nz=n, x0=0, y0=0, z0=0, lx=1.2, ly=1.2, lz=1.2,
             bcx=1, bcy=1, bcz=1, tend=0.05, nout=10**6, ndiag=10**6)
qf = field(rf, last_step(rf), (2*n, 2*n, 2*n))[:, n:, n:, n:]
qo = field(ro, last_step(ro), (n, n, n))
check(np.isfinite(qf).all() and np.isfinite(qo).all(), "finite fields")
check(qf[0].max() > 2.0, "blast wave formed (density jump)")
err = np.abs(qo[0] - qf[0]).max() / qf[0].max()
print(f"octant vs full max rel density difference: {err:.3e}")
check(err < 1e-8, "wall BC reproduces the mirror-symmetric solution")
print("PASS test_wall")
