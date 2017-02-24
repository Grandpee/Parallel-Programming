CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich

EXES                    := cuda openmp mpi

all: $(EXES)

clean:
	rm -f $(EXES)

block_FW.exe: block_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

seq_FW.exe: seq_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

cuda: All-Pair-Shortest-Path.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

openmp: All-Pair-Shortest-Path_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

mpi: All-Pair-Shortest-Path_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?


