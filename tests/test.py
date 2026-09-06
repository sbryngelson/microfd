# microcfd tests: python3 test.py [ic sod vortex wall mpi visc tgv3d]
import os, sys, glob, shutil, subprocess, pathlib, numpy as np
R = pathlib.Path(__file__).resolve().parent
MPIRUN = os.environ.get("MPIRUN", "mpirun --mca coll_hcoll_enable 0 --mca pml ucx -x UCX_TLS=^cuda_ipc").split()
ctr = lambda o, L, n: o + L / n * (np.arange(n) + 0.5)
def ok(c, m): print(("ok: " if c else "FAIL: ") + m); c or sys.exit(1)

def run(name, np_=1, **o):   # -> (diag rows [step t dt KE enstrophy maxMach ns/cell/step], first, last) (5,nz,ny,nx)
    d = R / "run" / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    p = subprocess.run(MPIRUN + ["-np", str(np_), str(R.parent / "microcfd")] + [f"{k}={v}" for k, v in o.items()],
                       cwd=d, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode: raise SystemExit(f"microcfd failed ({p.returncode}):\n{p.stdout}")
    diag = np.array([list(map(float, l.split())) for l in p.stdout.splitlines() if l and l[0].isdigit()])
    f, s = sorted(glob.glob(str(d / "out_*.bin"))) or [None, None], (5, *(o.get(k, 64) for k in ("nz", "ny", "nx")))
    return (diag, *(np.fromfile(b, np.float64).reshape(s) if b else None for b in (f[0], f[-1])))

def exact_rho(x, t, g=1.4, x0=0.5):              # exact Riemann density for Sod, ideal gas (Toro ch. 4)
    rl, ul, pl, rr, ur, pr = 1.0, 0.0, 1.0, 0.125, 0.0, 0.1
    cl, cr, p = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr), 0.5 * (pl + pr)
    def fk(p, rk, pk, ck):                       # pressure function and its derivative for one side
        A, B, w = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk, (p / pk) ** ((g - 1) / (2 * g))
        return ((p - pk) * np.sqrt(A / (p + B)), np.sqrt(A / (p + B)) * (1 - (p - pk) / (2 * (p + B)))) if p > pk \
            else (2 * ck / (g - 1) * (w - 1), (p / pk) ** (-(g + 1) / (2 * g)) / (rk * ck))
    for _ in range(100):                         # Newton iteration for the star pressure
        (fl, dl), (fr, dr) = fk(p, rl, pl, cl), fk(p, rr, pr, cr)
        p = max(p - (fl + fr + ur - ul) / (dl + dr), 1e-12)
    u = 0.5 * (ul + ur) + 0.5 * (fk(p, rr, pr, cr)[0] - fk(p, rl, pl, cl)[0])
    s = (np.asarray(x) - x0) / t; rho = np.empty_like(s)
    for sg, (rk, uk, pk, ck) in ((-1, (rl, ul, pl, cl)), (1, (rr, ur, pr, cr))):   # left/right of the contact
        m = s <= u if sg < 0 else s > u; sm = s[m]
        if p > pk:                               # shock
            rs = rk * (p / pk + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pk + 1)
            S = uk + sg * ck * np.sqrt((g + 1) / (2 * g) * p / pk + (g - 1) / (2 * g))
            rho[m] = np.where(sg * (sm - S) > 0, rk, rs)
        else:                                    # rarefaction, head at uk+sg*ck and tail at u+sg*cs
            rs, cs = rk * (p / pk) ** (1 / g), ck * (p / pk) ** ((g - 1) / (2 * g))
            fan = rk * (2 / (g + 1) - sg * (g - 1) / ((g + 1) * ck) * (uk - sm)) ** (2 / (g - 1))
            rho[m] = np.where(sg * (sm - uk - sg * ck) > 0, rk, np.where(sg * (sm - u - sg * cs) < 0, rs, fan))
    return rho

def ic():
    o = dict(case="tgv", nx=32, ny=32, nz=32, tend=0, nout=1)
    (d1, q1, _), (d2, q2, _) = run("ic1", 1, **o), run("ic2", 2, px=2, **o)
    dx = 2 * np.pi / 32; f = (np.sin(dx) / dx) ** 2   # central-difference factor in the discrete enstrophy
    ok(abs(d1[-1, 3] - 0.125) < 1e-12, "KE = 1/8")
    ok(abs(d1[-1, 4] - 0.375 * f) < 1e-9, "enstrophy = 3/8 (sin dx / dx)^2")
    ok(np.allclose(d1[-1, 3:5], d2[-1, 3:5], rtol=1e-12), "diagnostics agree on 2 ranks")
    ok(np.array_equal(q1, q2), "fields identical on 1 and 2 ranks")
    x = ctr(-np.pi, 2 * np.pi, 32); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")   # X along axis 0 = nx
    ok(np.allclose(q1[0], 1.0) and np.allclose(q1[1], (np.sin(X) * np.cos(Y) * np.cos(Z)).transpose(2, 1, 0),
                                               atol=1e-12), "rho and u match the analytic IC")
    xmf = (R / "run/ic1/out_000000.xmf").read_text()
    ok("out_000000.bin" in xmf and 'Dimensions="32 32 32"' in xmf, "XDMF written")
    print("PASS ic")

def sod():
    n, tol, rho, q0 = 200, float(os.environ.get("SOD_TOL", "1e-2")), {}, None
    for ax in range(3):
        N = [4, 4, 4]; N[ax] = n
        d, _, q = run(f"sod{ax}", 1, case="sod", axis=ax, nx=N[0], ny=N[1], nz=N[2], nout=10**6, ndiag=10**6)
        rho[ax] = np.moveaxis(q[0], 2 - ax, 0)[:, 0, 0]   # line along the shock-tube axis
        if ax == 0: q0 = q
        ok(np.isfinite(q).all(), f"axis {ax}: finite fields")
        ok(abs(d[-1, 1] - 0.2) < 1e-12, f"axis {ax}: reached t=0.2")
    err = np.abs(rho[0] - exact_rho(ctr(0, 1, n), 0.2)).mean(); print(f"Sod L1 density error at n={n}: {err:.4e}")
    ok(err < tol, f"L1 error {err:.3e} < {tol}")
    ok(max(np.abs(rho[i] - rho[0]).max() for i in (1, 2)) < 1e-12, "x, y, z axes agree")
    ok(np.array_equal(run("sod_np2", 2, case="sod", axis=0, nx=n, ny=4, nz=4, px=2, nout=10**6, ndiag=10**6)[2], q0),
       "1 rank == 2 ranks along x")
    print("PASS sod")

def vortex():
    e = {}
    for n in (32, 64, 128):
        _, a, b = run(f"vortex{n}", 1, case="vortex", nx=n, ny=n, nz=4, cfl=0.4, nout=10**6, ndiag=10**6)
        e[n] = np.abs(b[0] - a[0]).mean(); print(f"n={n:4d} L1(rho) = {e[n]:.3e}")
    p = np.log2(e[64] / e[128]); print(f"observed order 64->128: {p:.2f}")
    ok(p >= 1.9, "convergence order >= 1.9")   # dimension-by-dimension FV WENO5 is formally 2nd order in multi-D
    ok(e[32] > e[64] > e[128], "error decreases monotonically")
    print("PASS vortex")

def wall():
    n, o = 32, dict(case="sedov", mu=0.005, tend=0.05, nout=10**6, ndiag=10**6)
    qf = run("sedov_full", 1, nx=2*n, ny=2*n, nz=2*n, **o)[2][:, n:, n:, n:]
    qo = run("sedov_oct", 1, nx=n, ny=n, nz=n, x0=0, y0=0, z0=0, lx=1.2, ly=1.2, lz=1.2, bcx=1, bcy=1, bcz=1, **o)[2]
    ok(np.isfinite(qf).all() and np.isfinite(qo).all(), "finite fields")
    ok(qf[0].max() > 2.0, "blast wave formed (density jump)")
    err = np.abs(qo[0] - qf[0]).max() / qf[0].max(); print(f"octant vs full max rel density difference: {err:.3e}")
    ok(err < 1e-8, "wall BC reproduces the mirror-symmetric solution")
    print("PASS wall")

def mpi():
    o = dict(case="tgv", nx=64, ny=64, nz=64, tend=0.05, nout=10**6, ndiag=10**6)
    d1, _, q1 = run("mpi1", 1, **o); ok(d1[-1, 0] > 10, f"took {int(d1[-1,0])} steps")
    ok(np.array_equal(q1, run("mpi2", 2, px=1, py=2, pz=1, **o)[2]), "1 rank == 2 ranks (y split)")
    ok(np.array_equal(q1, run("mpi8", 8, **o)[2]), "1 rank == 8 ranks (2x2x2)")
    print("PASS mpi")

def visc():   # 2D TGV Re=10: KE decays as exp(-4 nu t) to 1 percent; 3D TGV at Re=0.1 is Stokes flow: exp(-6 nu t) to 0.2 percent
    d = run("tgv2d", 1, case="tgv2d", nx=64, ny=64, nz=4, nout=10**6, ndiag=10**6)[0]
    r, ex = d[-1, 3] / d[0, 3], np.exp(-0.4); print(f"2D KE(1)/KE(0) = {r:.5f}, exact {ex:.5f}"); ok(abs(r / ex - 1) < 0.01, "2D viscous decay")
    d = run("tgv3d_stokes", 1, case="tgv", mu=10, nx=64, ny=64, nz=64, tend=0.02, ndiag=10**6)[0]
    r, ex = d[-1, 3] / d[0, 3], np.exp(-1.2); print(f"3D KE ratio = {r:.5f}, exact {ex:.5f}"); ok(abs(r / ex - 1) < 2e-3, "3D Stokes decay")
    d = run("acoustic", 1, case="acoustic", nx=64, ny=4, nz=4, ndiag=10**6)[0]   # right-running wave, eps=1e-3, mu=0.05, Pr=0.71, t=4
    r, ex = d[-1, 3] / d[0, 3], np.exp(-(4 / 3 * 0.05 + 0.4 * 0.05 / 0.71) * 4)   # classical absorption: KE ~ exp(-k^2 [4/3 nu + (gamma-1) nu/Pr] t)
    print(f"acoustic KE ratio = {r:.5f}, classical {ex:.5f}"); ok(abs(r / ex - 1) < 0.01, "acoustic absorption: bulk stress and conduction"); print("PASS visc")

def tgv3d():   # 3D Taylor-Green Re=1600 at 128^3 vs the 512^3 spectral reference (minutes on one A100)
    ref = np.loadtxt(R / "ref/spectral_Re1600_512.gdiag")   # t, KE, -dKE/dt, enstrophy
    d = run("tgv3d", 1, case="tgv", nx=128, ny=128, nz=128, tend=10, nout=10**6, ndiag=20)[0]
    t, eps = d[:, 1], -np.gradient(d[:, 3], d[:, 1]); ip, ir = eps.argmax(), ref[:, 2].argmax()
    print(f"peak dissipation {eps[ip]:.5f} at t={t[ip]:.2f}; reference {ref[ir, 2]:.5f} at t={ref[ir, 0]:.2f}")
    ok(abs(eps[ip] / ref[ir, 2] - 1) < 0.10, "peak dissipation within 10 percent of the spectral reference")
    ok(abs(t[ip] - ref[ir, 0]) < 1.0, "peak time within 1.0")
    dev = np.abs(np.interp(t, ref[:, 0], ref[:, 2]) - eps).max(); print(f"max |eps - eps_ref| over [0,10]: {dev:.2e}")
    ok(dev < 2.5e-3, "dissipation curve within 2.5e-3 of the reference (128^3 dissipates early)"); print("PASS tgv3d")

def sedov():   # 64^3 blast to t=0.1: finite, positive, and mass conserved (nothing reaches the outflow boundary yet)
    _, a, b = run("sedov", 1, case="sedov", nx=64, ny=64, nz=64, nout=10**6, ndiag=10**6)
    ok(np.isfinite(b).all() and b[0].min() > 0 and b[4].min() > 0, "finite fields, positive density and pressure")
    ok(abs(b[0].sum() / a[0].sum() - 1) < 1e-10, "mass conserved"); print("PASS sedov")

def switches():   # every compile-time switch builds and passes Sod at a looser tolerance; the default build is restored
    mk = lambda x: subprocess.run(["make", "-B", "-C", str(R.parent), f"EXTRA={x}"], check=True, stdout=subprocess.DEVNULL)
    try:
        for x, tol in (("-DMUSCL", "2e-2"), ("-DRUSANOV", "1.5e-2"), ("-DHOST_MPI", "1e-2"), ("-DFLOAT", "1.5e-2")):
            mk(x)
            if x == "-DFLOAT":   # float32 output files: check the diagnostics only
                ok(np.isfinite(run("sod_float", 1, case="sod", nx=200, ny=4, nz=4, ndiag=10**6)[0]).all(), "FLOAT run finite"); continue
            p = subprocess.run([sys.executable, "test.py", "sod"], cwd=R, env=dict(os.environ, SOD_TOL=tol), text=True, capture_output=True)
            print(f"[{x}] " + p.stdout.strip().splitlines()[-1]); ok(p.returncode == 0, x)
    finally: mk("")
    print("PASS switches")

def perf():   # ns per cell per step on one GPU at 256^3, and weak scaling to 2 and 4 GPUs at 256^3 per GPU
    def ns(np_, **g):   # skip the warm-up rows and the partial last interval; column is per global cell, scale to per GPU
        d = run(f"perf{np_}", np_, case="tgv", tend=0.1, ndiag=10, **g)[0]; return d[2:-1, 6].mean() * np_, int(d[-1, 0])
    t1, steps = ns(1, nx=256, ny=256, nz=256); print(f"1 GPU, 256^3, {steps} steps: {t1:.2f} ns/cell/step = {1e3/t1:.0f} M cell-updates/s")
    t2, t4 = ns(2, nx=512, ny=256, nz=256, px=2)[0], ns(4, nx=512, ny=512, nz=256, px=2, py=2)[0]
    print(f"weak scaling, 256^3 per GPU: 1 GPU {t1:.2f}, 2 GPUs {t2:.2f} ({t1/t2*100:.0f}%), 4 GPUs {t4:.2f} ({t1/t4*100:.0f}%) ns/cell/step per GPU")
    print("reference: STREAmS-2 WENO5 14.2, MFC WENO5+HLLC normalized to 5 PDEs 8.9, goal < 7")
    ok(t1 < 12, "must-level target: below 12 ns per cell per step"); print("PASS perf")

for t in sys.argv[1:] or ["ic", "sod", "sedov", "vortex", "wall", "mpi", "visc", "tgv3d", "switches"]: globals()[t]()
