"""Shared helpers: run microcfd, parse diagnostics, read output fields."""
import os, glob, shutil, subprocess, pathlib, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXE = ROOT / "microcfd"
MPIRUN = os.environ.get(
    "MPIRUN", "mpirun --mca pml ob1 --mca btl smcuda,self,vader --mca coll_hcoll_enable 0").split()

def run(name, np_=1, **opts):
    """Run microcfd in tests/run/<name>; return (diagnostics array, run directory).

    Diagnostics columns: step t dt KE enstrophy maxMach ns_per_cell_step."""
    d = ROOT / "tests" / "run" / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    args = [f"{k}={v}" for k, v in opts.items()]
    r = subprocess.run(MPIRUN + ["-np", str(np_), str(EXE)] + args, cwd=d, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"microcfd failed ({r.returncode}):\n{r.stdout}")
    rows = [list(map(float, l.split())) for l in r.stdout.splitlines() if l and l[0].isdigit()]
    return np.array(rows), d

def last_step(rundir):
    return max(int(pathlib.Path(f).stem.split("_")[1]) for f in glob.glob(str(rundir / "out_*.bin")))

def field(rundir, step, N, dtype=np.float64):
    """Read out_<step>.bin as (5, nz, ny, nx): rho u v w p. N = (nx, ny, nz)."""
    nx, ny, nz = N
    return np.fromfile(rundir / f"out_{step:06d}.bin", dtype=dtype).reshape(5, nz, ny, nx)

def centers(o, L, n):
    return o + L / n * (np.arange(n) + 0.5)

def check(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)
    print("ok:", msg)
