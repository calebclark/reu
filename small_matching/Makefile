#DEBUG=1
#CPPFLAGS= -std=c++11 -lrt -O3 -Wall -Wextra -L/usr/local/cuda/lib64 -lcuda -lcudart 
BASICCPPFLAGS= -std=c++11 -Wall -Wextra 
BASICNVCCFLAGS=-std=c++11 -c -arch=sm_61 -I ~/cuba #-DTIME_KERNEL

ifdef DEBUG 
	CPPFLAGS=$(BASICCPPFLAGS) -D DEBUG -ggdb 
	NVCCFLAGS=$(BASICNVCCFLAGS) -D DEBUG -g -G -Xcompiler -rdynamic  
else
	CPPFLAGS=$(BASICCPPFLAGS) -O3 
	NVCCFLAGS=$(BASICNVCCFLAGS) -Xptxas -O3 
endif 


all: driver 
driver: cuda_test.o algs.o tests.o pii.o driver.cpp
	g++ $(CPPFLAGS) -c driver.cpp
	g++ $(CPPFLAGS) cuda_test.o pii.o driver.o algs.o tests.o -L/usr/local/cuda/lib64 -lcuda -lcudart -o driver
	
cuda_test.o: cuda_test.h cuda_test.cu
	nvcc $(NVCCFLAGS) -c cuda_test.cu
algs.o: algs.cpp algs.h 
	g++ $(CPPFLAGS) -c algs.cpp 
tests.o: tests.cpp tests.h
	g++ $(CPPFLAGS) -c tests.cpp
galgs.o: galgs.cu galgs.h
	nvcc $(NVCCFLAGS) -c galgs.cu
pii.o: pii.cu pii.h
	nvcc $(NVCCFLAGS) -c pii.cu

clean:
	rm -f driver *.o
