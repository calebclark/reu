UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    CXX=icpc
	CXXFLAGS= -qopenmp -std=c++11 -O3 -qopt-report -qopt-report-phase=vec
	LINKFLAGS= -qopenmp

all: stable-marriage-cpu stable-marriage-st stable-marriage-complete-cpu stable-marriage-complete-st
endif
ifeq ($(UNAME_S),Darwin)
    CXX=clang-omp++
	CXXFLAGS= -I. -g -std=c++11 -fopenmp -c -Wall -Wextra
	LINKFLAGS= -fopenmp

all: stable-marriage-cpu stable-marriage-st stable-marriage-complete-cpu stable-marriage-complete-st stable-marriage-complete-mic stable-marriage-mic
endif

genBipartiteGraph: genBipartiteGraph.o bipartiteIO.o manageMemory.o
	$(CXX) -o $@ $^ $(LINKFLAGS)
%.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

%.mic.o: %.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) -mmic
stable-marriage-cpu: manageMemory.o graphIO.o stableMarriageCpu.o
	$(CXX) -o $@ $^ $(LINKFLAGS)
stable-marriage-mic: manageMemory.mic.o graphIO.mic.o stableMarriageCpu.mic.o
	$(CXX) -o $@ $^ $(LINKFLAGS) -mmic
stable-marriage-st: manageMemory.o graphIO.o stableMarriageSinglethreaded.o
	$(CXX) -o $@ $^ $(LINKFLAGS)
stable-marriage-complete-cpu: manageMemory.o bipartiteIO.o stableMarriageCompleteCpu.o
	$(CXX) -o $@ $^ $(LINKFLAGS)
stable-marriage-complete-st:  manageMemory.o bipartiteIO.o stableMarriageCompleteSinglethreaded.o
	$(CXX) -o $@ $^ $(LINKFLAGS)
stable-marriage-complete-mic: manageMemory.mic.o bipartiteIO.mic.o stableMarriageCompleteCpu.mic.o
	$(CXX) -o $@ $^ $(LINKFLAGS) -mmic
clean:
	rm *.o stable-marriage-*