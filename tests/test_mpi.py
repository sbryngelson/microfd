"""Decomposition invariance: TGV on 1 rank vs 2 and 8 ranks must give identical fields."""
import numpy as np
from common import run, field, last_step, check

N = (64, 64, 64)
opts = dict(case="tgv", nx=64, ny=64, nz=64, tend=0.05, nout=10**6, ndiag=10**6)
d1, r1 = run("mpi1", 1, **opts)
d2, r2 = run("mpi2", 2, px=1, py=2, pz=1, **opts)
d8, r8 = run("mpi8", 8, **opts)
q1 = field(r1, last_step(r1), N)
check(d1[-1, 0] > 10, f"took {int(d1[-1,0])} steps")
check(np.array_equal(q1, field(r2, last_step(r2), N)), "1 rank == 2 ranks (y split)")
check(np.array_equal(q1, field(r8, last_step(r8), N)), "1 rank == 8 ranks (2x2x2)")
print("PASS test_mpi")
