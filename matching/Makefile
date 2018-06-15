#CPPFLAGS= -lrt -O3 -Wall -Wextra  
CPPFLAGS= -ggdb -Wall -Wextra 


all: driver algs tests
driver: algs tests driver.cpp
	g++ $(CPPFLAGS) -c driver.cpp
	g++ $(CPPFLAGS) driver.o algs.o tests.o -o driver
	

algs: algs.cpp algs.h 
	g++ $(CPPFLAGS) -c algs.cpp 
tests: tests.cpp tests.h
	g++ $(CPPFLAGS) -c tests.cpp 

clean:
	rm -f GS
