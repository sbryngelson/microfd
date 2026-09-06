# microcfd: GPU only, MPI always. `make` = NVIDIA, `make amd` = AMD.
CC     ?= mpicc
ARCH   ?= cc80
NVFLAGS  = -O3 -mp=gpu -gpu=$(ARCH),mem:separate
AMDFLAGS = -O3 -fopenmp --offload-arch=$(ARCH)
EXTRA  ?=

nvidia: microcfd.c
	OMPI_CC=nvc $(CC) $(NVFLAGS) $(EXTRA) $< -o microcfd -lm
amd: microcfd.c
	OMPI_CC=amdclang MPICH_CC=amdclang $(CC) $(AMDFLAGS) $(EXTRA) $< -o microcfd -lm
test: nvidia
	cd tests && for t in test_*.py; do python3 $$t || exit 1; done
clean:
	rm -rf microcfd tests/run *.bin *.xmf
.PHONY: nvidia amd test clean
