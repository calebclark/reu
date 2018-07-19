DEBUG=1
BASICCFLAGS= -Wall -Wextra  -std=gnu11 #-D PRINT_FAILS

ifdef DEBUG 
	CFLAGS=$(BASICCFLAGS) -D DEBUG -ggdb 
else
	CFLAGS=$(BASICCFLAGS) -O3 
endif 


all: graphs
graphs: graphs.c algs.o
	gcc $(CFLAGS) algs.c graphs.c -o graphs
algs.o: algs.c algs.h
	gcc $(CFLAGS) -c algs.c


clean:
	rm -f graphs *.o *.png
	
