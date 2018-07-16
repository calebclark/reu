DEBUG=1
BASICCFLAGS= -Wall -Wextra  -std=gnu11

ifdef DEBUG 
	CFLAGS=$(BASICCFLAGS) -D DEBUG -ggdb 
else
	CFLAGS=$(BASICCFLAGS) -O3 
endif 


all: graphs
graphs: graphs.c
	gcc $(CFLAGS) graphs.c -o graphs

clean:
	rm -f graphs *.o *.png
	
