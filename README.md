# microcfd

A 3D compressible Navier-Stokes solver in one C file. Finite volume on a uniform
grid, WENO5-Z reconstruction, HLLC flux, SSP-RK3, 2nd-order viscous terms
(walls are reflective free-slip, adiabatic).
Dimension-by-dimension with one Riemann solve per face, so formally 2nd order in
multi-D with a WENO5 error constant (about 10x below MUSCL). Runs on NVIDIA and
AMD GPUs through OpenMP target offload, decomposed with MPI. Dependencies: a C
compiler with OpenMP offload and MPI. Tests use Python + numpy.

## Build

```
make            # NVIDIA: nvc, ARCH=cc80 default
make amd ARCH=gfx90a
```

Compile-time switches via `EXTRA` (use `make -B` to force the rebuild):
`-DMUSCL` (van Leer instead of WENO5-Z), `-DRUSANOV` (instead of HLLC),
`-DFLOAT` (single precision), `-DHOST_MPI` (stage halos through host memory for
non-GPU-aware MPI).

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
| ndiag | steps between diagnostics lines (must be > 0) | 10 |
| nout | steps between field outputs, 0 = never | 0 |
| axis | shock-tube axis for sod | 0 |

Diagnostics go to stdout: `step t dt KE enstrophy maxMach ns_per_cell_step`
(mean KE and enstrophy per cell). Fields go to `out_NNNNNN.bin` as
`[5][nz][ny][nx]` doubles (rho, u, v, w, p; float32 under `-DFLOAT`) with an
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
rebuilds the binary five times, four switches and the default restore).
If `switches` is interrupted, run `make -B` to restore the default build.
`python3 test.py perf` reports throughput and weak scaling on up to 4 GPUs.

## Performance

TGV, 256^3, WENO5-Z + HLLC + viscous, double; microcfd rows measured on one
A100 80GB PCIe and one MI210 respectively:

| code | ns per cell per step | source |
|---|---|---|
| microcfd | 4.69 | python3 test.py perf |
| microcfd, MI210 (gfx90a), user-measured | 4.46 | make amd |
| STREAmS-2, WENO5 | 14.2 | Sathyanarayana et al. 2023, A100 40GB |
| MFC, WENO5 + HLLC, normalized to 5 PDEs | 8.9 | Wilfong et al. 2024 |

Weak scaling to 4 GPUs: 1 GPU 4.69, 2 GPUs 5.07 (93%), 4 GPUs 6.16 (76%) ns/cell/step per GPU.
