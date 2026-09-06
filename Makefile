# microfd: MPI always. `make` = NVIDIA, `make amd` = AMD, `make cpu` = host (no offload).
MPICC  ?= mpicc
ARCH   ?= cc80
NVFLAGS  = -O3 -mp=gpu -gpu=$(ARCH),mem:separate
# assume-* keep the face kernel in SPMD mode; without them clang emits Generic-SPMD at 64 threads (30x slower)
AMDFLAGS = -O3 -fopenmp --offload-arch=$(ARCH) -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state
CPUFLAGS = -O2 -fopenmp                 # no offload: target regions run on the host
EXTRA  ?=
MK     ?= nvidia

nvidia: microfd.c
	OMPI_CC=nvc $(MPICC) $(NVFLAGS) $(EXTRA) $< -o microfd -lm
amd: microfd.c
	OMPI_CC=amdclang MPICH_CC=amdclang $(MPICC) $(AMDFLAGS) $(EXTRA) $< -o microfd -lm
cpu: microfd.c
	$(MPICC) $(CPUFLAGS) $(EXTRA) $< -o microfd -lm
test:
	$(MAKE) $(MK)
	cd tests && python3 test.py
clean:
	rm -rf microfd tests/run *.bin *.xmf
.PHONY: nvidia amd cpu test clean
