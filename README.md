# microfd

[![ci](https://github.com/sbryngelson/microfd/actions/workflows/ci.yml/badge.svg)](https://github.com/sbryngelson/microfd/actions/workflows/ci.yml)
![Lines of Code](sloc.svg)
![C11](https://img.shields.io/badge/C11-single%20file-blue)
![OpenMP](https://img.shields.io/badge/OpenMP-target%20offload-orange)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD-lightgrey)

How short can a very fast CFD code be? microfd is a 3D compressible Navier-Stokes solver in one short C file: 1.2 ns per cell per step on one MI350X, 3.9 on one A100.

![Taylor-Green vortex at Re 1600, 256^3](tgv.webp)

Finite volume, WENO5-Z, HLLC, SSP-RK3, viscous terms; OpenMP offload to NVIDIA and AMD GPUs; MPI across GPUs.

## Build and run
```
make                     # NVIDIA
make amd ARCH=gfx90a     # AMD
make cpu                 # host, no offload: same answers, slowly
mpirun -np 4 ./microfd case=tgv nx=256 ny=256 nz=256 tend=10 nout=500
make test
```

| key | meaning | default |
|---|---|---|
| case | tgv, tgv2d, sod, sedov, vortex, acoustic | tgv |
| nx ny nz | cells | 64 |
| px py pz | ranks per direction | auto |
| bcx bcy bcz | 0 periodic, 1 wall, 2 outflow | case |
| mu, pr, gamma | viscosity, Prandtl, gamma | case, 0.71, 1.4 |
| cfl, tend | CFL number, end time | 0.5, case |
| ndiag, nout | steps between diagnostics, field outputs | 10, never |

Output: `out_NNNNNN.bin` with `rho u v w p` as `[5][nz][ny][nx]` doubles, plus an `.xmf` ParaView opens.

`make cpu` drops `--offload-arch`, so the `target` regions run on the host and give the same answers. CI builds that way and runs `ic sod wall`, so a green badge covers the numerics and the MPI decomposition, not the offload path.

## Performance
Taylor-Green, viscous, double precision, ns per cell per step:

| | GPU | grid | ns |
|---|---|---|---|
| microfd | MI350X | 976^3 | 1.20 |
| microfd | MI300X | 256^3 | 1.41 |
| microfd | A100 80GB | 256^3 | 3.92 |
| microfd | MI250X, one GCD | 592^3 | 4.51 |
| microfd | MI210 | 576^3 | 4.55 |
| MFC (Wilfong et al. 2024) | A100 | 8M cells | 8.9 |
| STREAmS-2 (Sathyanarayana et al. 2023) | A100 40GB | 33.6M points | 14.2 |

## License
Apache-2.0
