#include"memoryManager.h"
#include"bipartiteIO.h"
#include"assert.h"
#include"stdio.h"
#include"stdlib.h"

int main(int argc, char**argv) {


    unsigned int n = 1;
    printf("\nEnter n: ");
    int retCode = scanf("%u", &n);

    //Allocate Graph on host
    CompleteBipartiteGraph *hostgraph = allocateComBipartite(n, n);

    //write generated Graph on host
    char *name = writeRanks(hostgraph);

    free(name);
    return 0;
}
