#DEBUG=1
BASICCFLAGS= -Wall -Wextra 

ifdef DEBUG 
	CFLAGS=$(BASICFLAGS) -D DEBUG -ggdb 
else
	CFLAGS=$(BASICFLAGS) -O3 
endif 


all: graphs
graphs: graphs.c
	gcc $(CFLAGS) graphs.c -o graphs

clean:
	rm -f graphs *.o
