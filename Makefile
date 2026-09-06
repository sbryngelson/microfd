# microcfd: GPU only, MPI always. `make` = NVIDIA, `make amd` = AMD.
MPICC  ?= mpicc
ARCH   ?= cc80
NVFLAGS  = -O3 -mp=gpu -gpu=$(ARCH),mem:separate
AMDFLAGS = -O3 -fopenmp --offload-arch=$(ARCH)
EXTRA  ?=

nvidia: microcfd.c
	OMPI_CC=nvc $(MPICC) $(NVFLAGS) $(EXTRA) $< -o microcfd -lm
amd: microcfd.c
	OMPI_CC=amdclang MPICH_CC=amdclang $(MPICC) $(AMDFLAGS) $(EXTRA) $< -o microcfd -lm
test: nvidia
	cd tests && python3 test.py
clean:
	rm -rf microcfd tests/run *.bin *.xmf
.PHONY: nvidia amd test clean
