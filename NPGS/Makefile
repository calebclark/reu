CPPFLAGS= -std=c++11 -lrt -O3 -Wall -Wextra  
#CPPFLAGS= -std=c++11 -D DEBUG -ggdb -Wall -Wextra 
NVCCFLAGS=-std=c++11 -c -arch=sm_61 -c GScuda.cu 
 


all: alln driver
driver: algs.o tests.o driver.cpp
	g++ $(CPPFLAGS) -c driver.cpp
	g++ $(CPPFLAGS) driver.o algs.o tests.o -o driver

alln: algs.o tests.o alln.cpp
	g++ $(CPPFLAGS) -c alln.cpp
	g++ $(CPPFLAGS) alln.o algs.o tests.o -o alln
	

algs.o: algs.cpp algs.h 
	g++ $(CPPFLAGS) -c algs.cpp 
tests.o: tests.cpp tests.h
	g++ $(CPPFLAGS) -c tests.cpp
galgs.o: galgs.cu galgs.h
	nvcc $(NVCCFLAGS) -c galgs.cu

clean:
	rm -f driver *.o
