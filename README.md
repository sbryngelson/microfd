# microcfd

A 3D compressible Navier-Stokes solver in one C file. Finite volume on a uniform
grid, WENO5-Z reconstruction, HLLC flux, SSP-RK3, 2nd-order viscous terms
(walls are reflective free-slip, adiabatic).
Dimension-by-dimension with one Riemann solve per face, so formally 2nd order in
multi-D with a WENO5 error constant (about 10x below a 2nd-order limiter). Runs on NVIDIA and
AMD GPUs through OpenMP target offload, decomposed with MPI. Dependencies: a C
compiler with OpenMP offload and MPI. Tests use Python + numpy.

## Build

```
make            # NVIDIA: nvc, ARCH=cc80 default
make amd ARCH=gfx90a
```

One compile-time switch via `EXTRA` (use `make -B` to force the rebuild):
`-DHOST_MPI` stages halos through host memory for MPI that is not GPU-aware.
Precision is double throughout.

## Run

```
mpirun --mca coll_hcoll_enable 0 --mca pml ucx -x UCX_TLS=^cuda_ipc -np 4 ./microcfd case=tgv nx=256 ny=256 nz=256 tend=10 ndiag=20 nout=500
```

Options are `key=value`. Unknown keys are an error.

| key | meaning | default |
|---|---|---|
| case | tgv, tgv2d, sod, sedov, vortex, acoustic | tgv |
| nx ny nz | global cells | 64 |
| px py pz | ranks per direction, 0 = automatic | 0 |
| x0 y0 z0, lx ly lz | domain origin and lengths | case |
| bcx bcy bcz | 0 periodic, 1 wall, 2 outflow | case |
| gamma, mu, pr | gas constants; mu=0 gives Euler | 1.4, case, 0.71 |
| cfl, tend | CFL number, end time | 0.5, case |
| ndiag | steps between diagnostics lines | 10 |
| nout | steps between field outputs, 0 = never | 0 |
| axis | shock-tube axis for sod | 0 |

Diagnostics go to stdout: `step t dt KE enstrophy maxMach ns_per_cell_step`
(mean KE and enstrophy per cell). Fields go to `out_NNNNNN.bin` as
`[5][nz][ny][nx]` doubles (rho, u, v, w, p) with an
`out_NNNNNN.xmf` wrapper that ParaView opens directly.

On this machine (A100 PCIe, NVIDIA HPC SDK 25.11 HPC-X), direct GPU-to-GPU
copies are corrupt whenever peer access is enabled, so any MPI transport that
uses CUDA IPC returns garbage. Launch with
`mpirun --mca coll_hcoll_enable 0 --mca pml ucx -x UCX_TLS=^cuda_ipc`, which
stages device buffers through pinned host memory inside UCX. Set `MPIRUN` to
override what the tests use; `-DHOST_MPI` is the fallback for an MPI that is
not GPU-aware at all.

## Tests

`make test` runs `tests/test.py` (`tgv3d` takes a few minutes; `switches`
rebuilds the binary twice, one switch and the default restore).
If `switches` is interrupted, run `make -B` to restore the default build.
`python3 test.py perf` reports throughput and weak scaling on up to 4 GPUs.

## Performance

TGV, WENO5-Z + HLLC + viscous, double precision; one full SSP-RK3 step. Grid
size matters: 256^3 is a few percent of a modern GPU, so both a fill-size and a
256^3 row are given. The published codes are quoted at the sizes they reported.

| code | GPU | grid | ns per cell per step | source |
|---|---|---|---|---|
| microcfd | MI350X (gfx950) | 976^3, 930M cells, 212 of 287 GiB | 1.20 | `make amd ARCH=gfx950` |
| microcfd | MI350X (gfx950) | 256^3, 16.8M cells | 1.51 | as above |
| microcfd | MI210 (gfx90a) | 576^3, 191M cells, 44 of 64 GiB | 4.55 | `make amd ARCH=gfx90a` |
| microcfd | MI210 (gfx90a) | 256^3, 16.8M cells | 4.77 | as above |
| microcfd | A100 80GB PCIe | 256^3, 16.8M cells | 4.69 | `python3 test.py perf` |
| STREAmS-2, WENO5 | A100 40GB | 33.6M points | 14.2 | Sathyanarayana et al. 2023 |
| MFC, WENO5 + HLLC, normalized to 5 PDEs | A100 | 8M cells | 8.9 | Wilfong et al. 2024 |

Strong scaling on MI350X at 976^3: 1 GPU 1.20, 2 GPUs 0.65 (92%), 4 GPUs 0.34
(88%). The 8-GPU point measures 0.15, which is superlinear against that trend
and is not yet confirmed.

Weak scaling on A100 to 4 GPUs: 1 GPU 4.69, 2 GPUs 5.07 (93%), 4 GPUs 6.16
(76%) ns/cell/step per GPU. The A100 rows predate the fused divergence/update
kernel and have not been re-run; expect them to improve by about 15%.

Memory is 240 B per cell (30 fields of 8 B: q, q1, w, and F for three
directions), so a 64 GiB GPU holds roughly 630^3 and a 287 GiB GPU roughly 1050^3.
