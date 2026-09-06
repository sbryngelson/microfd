"""Initial condition, diagnostics, and MPI-IO output on 1 and 2 ranks (tend=0, no time stepping)."""
import numpy as np
from common import run, field, centers, check

N = (32, 32, 32)
opts = dict(case="tgv", nx=32, ny=32, nz=32, tend=0, nout=1)

d1, r1 = run("ic1", 1, **opts)
d2, r2 = run("ic2", 2, px=2, **opts)

# kinetic energy of the TGV is exactly 1/8 on any uniform grid; enstrophy 3/8 times the central-difference factor
dx = 2 * np.pi / 32
f = (np.sin(dx) / dx) ** 2
check(abs(d1[-1, 3] - 0.125) < 1e-12, "KE = 1/8")
check(abs(d1[-1, 4] - 0.375 * f) < 1e-9, "enstrophy = 3/8 (sin dx / dx)^2")
check(np.allclose(d1[-1, 3:5], d2[-1, 3:5], rtol=1e-12), "diagnostics agree on 2 ranks")

q1, q2 = field(r1, 0, N), field(r2, 0, N)
check(np.array_equal(q1, q2), "fields identical on 1 and 2 ranks")
x = centers(-np.pi, 2 * np.pi, 32)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")          # X varies along axis 0 = nx; field is (nz,ny,nx)
u = (np.sin(X) * np.cos(Y) * np.cos(Z)).transpose(2, 1, 0)
check(np.allclose(q1[0], 1.0) and np.allclose(q1[1], u, atol=1e-12), "rho and u match the analytic IC")
xmf = (r1 / "out_000000.xmf").read_text()
check("out_000000.bin" in xmf and 'Dimensions="32 32 32"' in xmf, "XDMF written")
print("PASS test_ic")
