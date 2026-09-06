"""Isentropic vortex advected one period: L1 density error must converge at order >= 1.9 between 64 and 128 cells (dimension-by-dimension FV WENO is formally 2nd order in multi-D; the WENO5 reconstruction lowers the constant)."""
import numpy as np
from common import run, field, last_step, check

errs = {}
for n in (32, 64, 128):
    N = (n, n, 4)
    d, r = run(f"vortex{n}", 1, case="vortex", nx=n, ny=n, nz=4, cfl=0.4, nout=10**6, ndiag=10**6)
    q0, q1 = field(r, 0, N), field(r, last_step(r), N)
    errs[n] = np.abs(q1[0] - q0[0]).mean()
    print(f"n={n:4d} L1(rho) = {errs[n]:.3e}")
p = np.log2(errs[64] / errs[128])
print(f"observed order 64->128: {p:.2f}")
check(p >= 1.9, "convergence order >= 1.9")
check(errs[32] > errs[64] > errs[128], "error decreases monotonically")
print("PASS test_vortex")
