# microcfd: GPU only, MPI always. `make` = NVIDIA, `make amd` = AMD.
MPICC  ?= mpicc
ARCH   ?= cc80
NVFLAGS  = -O3 -mp=gpu -gpu=$(ARCH),mem:separate
# assume-* keep the face kernel in SPMD mode; without them clang emits Generic-SPMD at 64 threads (30x slower)
AMDFLAGS = -O3 -fopenmp --offload-arch=$(ARCH) -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state
EXTRA  ?=
MK     ?= nvidia

nvidia: microcfd.c
	OMPI_CC=nvc $(MPICC) $(NVFLAGS) $(EXTRA) $< -o microcfd -lm
amd: microcfd.c
	OMPI_CC=amdclang MPICH_CC=amdclang $(MPICC) $(AMDFLAGS) $(EXTRA) $< -o microcfd -lm
test:
	$(MAKE) $(MK)
	cd tests && python3 test.py
clean:
	rm -rf microcfd tests/run *.bin *.xmf
.PHONY: nvidia amd test clean
