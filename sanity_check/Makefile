#CPPFLAGS= -std=c++11 -lrt -O3 -Wall -Wextra  
CPPFLAGS= -std=c++11 -O3 -ggdb -Wall -Wextra 
NVCCFLAGS=-std=c++11 -c -arch=sm_61 -c GScuda.cu 
 


all: driver 
driver: algs.o tests.o driver.cpp
	g++ $(CPPFLAGS) -c driver.cpp
	g++ $(CPPFLAGS) driver.o algs.o tests.o -o driver
	

algs.o: algs.cpp algs.h 
	g++ $(CPPFLAGS) -c algs.cpp 
tests.o: tests.cpp tests.h
	g++ $(CPPFLAGS) -c tests.cpp

clean:
	rm -f driver *.o
