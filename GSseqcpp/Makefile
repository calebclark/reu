CPPFLAGS= -lrt -O3 -Wall -Wextra  
#CFLAGS= -ggdb -Wall -Wextra -std=c99 


all: GS tests

GS: GS.cpp 
	g++ $(CPPFLAGS) GS.cpp -o GS
tests: tests.cpp tests.h
	g++ $(CPPFLAGS) -c tests.cpp 

clean:
	rm -f GS
