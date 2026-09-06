# microcfd design

Date: 2026-09-05
Status: draft for review

## Goal

A 3D compressible Navier-Stokes solver that is fast, runtime-configurable,
and very short. One C file. Runs on NVIDIA and AMD GPUs and on CPUs from the
same source. Dependencies: a C compiler with OpenMP 5 target offload and an
MPI library. Nothing else.

Line budget: about 600 lines of C including MPI and I/O.

## Non-goals

- Multiphase, multicomponent, reacting, or real-gas physics.
- Non-uniform or curvilinear grids, AMR, immersed boundaries.
- Implicit time stepping.
- Any third-party numerics, I/O, or portability library.

## Decisions already made

| Question | Decision |
|---|---|
| Language | C11, single file `microcfd.c` |
| Offload | OpenMP 5 `target teams loop` and `target data`. No vendor API calls. |
| Generic | Runtime-configurable problems. Fixed 5-equation Navier-Stokes, ideal gas. |
| Physics | Navier-Stokes with constant viscosity and Prandtl number. Euler is `mu=0`. |
| Precision | `typedef double real`, `-DFLOAT` switches to single. |
| Parallelism | MPI 3D Cartesian decomposition, one rank per GPU, GPU-aware halo exchange. |
| Reference perf | Published numbers, not reruns of other codes. See section 8. |

## 1. Toolchain

One Makefile, one source, two build lines. GPU only, MPI always:

```
mpicc (nvc)       -O3 -mp=gpu -gpu=cc80,mem:separate        # NVIDIA
mpicc (amdclang)  -O3 -fopenmp --offload-arch=gfx90a        # AMD
```

On this machine the MPI is the HPC-X Open MPI bundled with the NVIDIA HPC
SDK. Its UCX path silently corrupts device-buffer messages here, while the
`ob1` PML with the `smcuda` BTL exchanges device buffers correctly. Runs use:

```
mpirun --mca pml ob1 --mca btl smcuda,self,vader --mca coll_hcoll_enable 0
```

Only constructs that both nvc and amdclang handle well are used:
`target enter data`, `target exit data`, `target teams loop collapse(3)`,
`target teams loop reduction(max:)`, `target data use_device_ptr`.
No `declare target` functions with recursion, no device-side malloc,
no atomics in hot loops.

Verification platforms: 4x A100 80GB PCIe here (NVIDIA HPC SDK 25.11), and
the user's AMD machine for the ROCm build. There is no CPU build.

## 2. Data layout

Structure of arrays. Each field is a contiguous 3D block with 3 ghost layers
on every side (WENO5 stencil). x is fastest. A single flat allocation per
array holds 5 fields.

```
#define NG 3
#define I(v,i,j,k) ((((size_t)(v)*(nz+2*NG) + (k))*(ny+2*NG) + (j))*(nx+2*NG) + (i))
```

Device-resident arrays per rank, 5 fields each:

| Array | Contents | Lifetime |
|---|---|---|
| q   | conserved: rho, rho u, rho v, rho w, E | whole run |
| q1  | RK stage state | whole run |
| rhs | accumulated flux divergence | whole run |
| w   | primitives: rho, u, v, w, p | whole run |
| F   | face fluxes for the current direction | whole run, reused |

Plus six halo send and six receive buffers, sized for the largest face
times 5 fields times NG layers.

Memory per cell: 25 doubles = 200 B. A 512^3 block per rank is 27 GB.

## 3. Time step algorithm

Per step:

1. `dt` kernel: device max reduction of `(|u|+c)/dx` over three directions
   plus the viscous limit `nu/dx^2`, then `MPI_Allreduce`. Once per step.
2. Three SSP-RK3 stages. Stage s calls `rhs_eval(qin)` then `update`.

SSP-RK3 in the two-register form:

```
q1 = q  + dt L(q)
q1 = 3/4 q + 1/4 q1 + 1/4 dt L(q1)
q  = 1/3 q + 2/3 q1 + 2/3 dt L(q1)
```

`update(qout, a, qa, b, qb, c)` computes `qout = a*qa + b*qb + c*dt*rhs`
and zeroes `rhs` in the same pass, so no separate memset.

`rhs_eval(qin)`:

1. `halo(qin)` for x, then y, then z. Sequential order fills edges and
   corners through the face exchanges, so only 6 messages are needed.
2. `prim(qin, w)` over the full padded block including ghosts.
3. For each direction d in {x, y, z}, with stride `s_d` and spacing `h_d`:
   - `face(w, F, s_d, h_d)`: for each face i+1/2 in the interior plus one
     layer, reconstruct left and right primitive states with WENO5-Z from the
     6-cell stencil along d, compute the HLLC flux, subtract the viscous and
     heat flux at the face, write to F.
   - `div(F, rhs, s_d, h_d)`: `rhs -= (F[i] - F[i-1]) / h_d`.

Six kernel bodies total: dt, halo pack/unpack, prim, face, div, update.
One `face` function serves all three directions through the stride argument.

### Reconstruction and flux

- WENO5-Z on primitive variables, no characteristic projection. Component-wise
  primitive WENO is a known, adequate choice at this length budget.
- HLLC with Davis wave-speed estimates.
- Viscous flux at the face: normal derivatives directly from the two
  adjacent cells. Tangential derivatives by averaging the central
  differences of the two adjacent cells. This needs diagonal neighbors,
  which the sequential halo exchange provides. Stress tensor with Stokes
  hypothesis, heat flux `-k dT/dn` with `k = cp mu / Pr`.
- Compile-time alternatives, each under 15 lines: `-DMUSCL` (van Leer
  limiter) replaces WENO5-Z, `-DRUSANOV` replaces HLLC.

### Boundary conditions

Chosen per axis at runtime as an integer: 0 periodic, 1 wall (reflective,
adiabatic), 2 outflow (zero gradient). Periodic is handled entirely by the Cartesian
communicator. Wall and outflow are applied by a ghost-fill kernel on ranks
whose neighbor in that direction is `MPI_PROC_NULL`.

## 4. MPI

- `MPI_Cart_create` with dims from input or `MPI_Dims_create`, periodicity
  from the BC choice per axis. `MPI_Cart_shift` gives the six neighbors.
- Pack kernel copies NG layers of 5 fields into a contiguous device buffer.
  `MPI_Isend` and `MPI_Irecv` on device pointers inside
  `#pragma omp target data use_device_ptr(...)`. Unpack kernel writes the
  received layers into the ghost region.
- `-DHOST_MPI` stages buffers through host memory with `target update` for
  MPI builds that are not GPU-aware.
- One rank per GPU. Device selection by `rank % omp_get_num_devices()`.
- Global reductions: one `MPI_Allreduce` per step for dt, plus diagnostics at
  output intervals.

## 5. Input

`key=value` arguments on the command line, parsed with `sscanf`. Unknown keys
are an error.

| Key | Meaning | Default |
|---|---|---|
| case | tgv, tgv2d, sod, sedov, vortex | tgv |
| nx ny nz | global cells | 64 64 64 |
| lx ly lz | domain lengths | 2pi each |
| px py pz | rank decomposition, 0 means auto | 0 |
| gamma | ratio of specific heats | 1.4 |
| mu | dynamic viscosity, 0 means Euler | case default |
| pr | Prandtl number | 0.71 |
| cfl | CFL number | 0.5 |
| tend | end time | case default |
| bcx bcy bcz | 0 periodic, 1 wall, 2 outflow | case default |
| nout | steps between outputs | 0 means never |
| ndiag | steps between diagnostics lines | 10 |

Each built-in case is one C function of about 8 lines that sets the
primitive state at a point and the case defaults. Sod takes an axis via
`axis=x|y|z`. Adding a case means adding one such function and one line in
the dispatch table.

## 6. Output

- Diagnostics to stdout every `ndiag` steps: step, t, dt, total kinetic
  energy, enstrophy (TGV), max Mach, and wall time per step.
- Fields every `nout` steps: one raw binary file for the whole domain,
  written collectively with `MPI_File_write_at_all` using a subarray type.
  Layout is `[5][nz][ny][nx]` double, little-endian, no header.
- A companion XDMF text file per output so ParaView opens the raw file
  directly. About 15 lines of `fprintf` on rank 0.

## 7. Testing

Test drivers are Python scripts under `tests/` that run the binary and check
outputs. Python is a test dependency only. The code has no Python dependency.

| Test | Check | Pass criterion |
|---|---|---|
| Sod x, y, z | density vs exact Riemann solution at t=0.2 | L1 error below fixed threshold, identical across axes to roundoff |
| Isentropic vortex | L2 error vs exact after one period, at 32, 64, 128 cells | observed order at least 3 |
| 2D Taylor-Green, Re 10, Ma 0.05 | kinetic energy decay vs exact exp(-4 nu t) at t=1 | within 1 percent |
| TGV Re 1600, Ma 0.1 | kinetic energy dissipation rate vs the 512^3 spectral reference at 128^3 | peak dissipation within 10 percent, peak time within 0.6 |
| Wall BC | Sedov in one octant with walls vs full domain | octant matches the full run's octant to 1e-8 |
| MPI | TGV 64^3 on 1 rank vs 8 ranks for 50 steps | fields agree to roundoff |
| Perf | TGV 256^3 on 1 A100, 100 steps | ns per cell per step, reported, compared to section 8 |
| Weak scaling | TGV 256^3 per rank on 1, 2, 4 A100s | efficiency reported |

Each numerical test is added before the feature it exercises, per the
test-driven workflow in the implementation plan.

## 8. Performance target

Metric: nanoseconds per cell per time step on one A100, full RK3 step,
WENO5 plus HLLC plus viscous terms, double precision. Lower is better.

Published state of the art for comparable GPU compressible Navier-Stokes
codes on one A100:

| Code | Setup | A100 result | Source |
|---|---|---|---|
| STREAmS-2 | CUDA Fortran, WENO5 everywhere, RK3, 33.6M points, A100 40GB | 0.476 s per step = 14.2 ns per point per step | Sathyanarayana et al. 2023, Table 3 |
| STREAmS-2 | same, 6th-order central instead of WENO | 0.318 s per step = 9.5 ns per point per step | same |
| MFC | OpenACC Fortran, WENO5, HLLC, RK3, two-phase 5-equation model, 8M cells, A100 PCIe | 0.59 ns per cell per PDE per RHS evaluation. For 7 PDEs and 3 RHS evaluations that is about 12.4 ns per cell per step. Normalized to 5 PDEs, about 8.9 ns. | Wilfong et al. 2024, Fig. 6 |
| STREAmS-2 on MI250X GCD | as above, WENO5 | 29.3 ns per point per step | Sathyanarayana et al. 2023 |
| MFC on MI250X GCD | as above | 1.09 ns per cell per PDE per RHS | Wilfong et al. 2024 |

Sources:
- https://arxiv.org/abs/2304.05494
- https://arxiv.org/abs/2409.10729

Roofline for this design on the A100 80GB PCIe (1.94 TB/s HBM, 9.7 TFLOP/s
FP64): the design moves about 880 B per cell per RK stage and does roughly
4000 flops per cell per stage. That is about 1.8 ns per cell per step at
achievable bandwidth and about 1.2 ns at peak FP64. WENO5 arithmetic will
dominate in practice, so 3 to 5 ns per cell per step is the realistic floor.

Targets:

| Level | ns per cell per step, A100, WENO5 + HLLC + NS, FP64 | Meaning |
|---|---|---|
| Must | below 12 | beats STREAmS-2 WENO5 and MFC per-cell like for like |
| Goal | below 7 | roughly 2x STREAmS-2 WENO5, about 140M cell updates per second |
| MUSCL build | below 4 | shows the code is bandwidth-bound once WENO arithmetic is removed |

Caveat: the A100 here is the 80GB PCIe part with about 25 percent more HBM
bandwidth than the 40GB card STREAmS-2 used. The Goal level accounts for
that. Results are reported with both raw and bandwidth-normalized numbers.

Fallback: if OpenMP offload measures more than 1.5x slower than the roofline
estimate after the obvious fixes (collapse order, launch bounds, register
pressure), the face kernel is ported to a CUDA/HIP source with a 20-line
macro shim. That decision is made from measurements, not up front.

## 9. File layout

```
microcfd.c        solver, ~600 lines
Makefile          three build lines, test target
tests/            Python drivers and reference data
README.md         build, run, cases, perf table
docs/superpowers/specs/   this document
```

## 10. Risks

- OpenMP offload code generation quality for the WENO face kernel on
  amdclang. Mitigation: measure early on the AMD machine, keep the CUDA/HIP
  shim fallback scoped to one kernel.
- Register pressure in the fused face kernel (WENO5 on 5 variables, HLLC,
  viscous terms). Mitigation: measure occupancy, split viscous flux into a
  second face pass if needed. Costs one extra F read and write.
- GPU-aware MPI availability. Mitigation: `-DHOST_MPI`.
- Primitive-variable WENO can produce small oscillations near strong shocks.
  Acceptable for this scope. Characteristic projection is out of budget.

## 11. Code style

Compactness is a first-class goal and cleverness in service of it is
welcome. Concretely:

- Prefer one generic function with a stride or coefficient argument over
  three near-copies. The face, divergence, pack, unpack, and ghost-fill
  kernels are all direction-generic through strides.
- Prefer macros that generate repeated per-field code over hand-written
  repetition, as long as the macro is defined once and named clearly.
- Prefer folding a pass into an adjacent kernel over a separate kernel,
  such as zeroing the residual inside the update kernel.
- Prefer table-driven dispatch (cases, boundary conditions, RK coefficients)
  over if-else chains.
- Dense is fine. Obscure is not. Every trick that saves lines must still be
  readable by someone who knows finite-volume CFD, given at most a one-line
  comment.
- No dead code, no configurability that is not used by a test or a case.
