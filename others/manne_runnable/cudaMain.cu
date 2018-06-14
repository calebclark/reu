#include <stdlib.h>
#include <stdio.h>

#include"hostdriver.h"
#include"graphIO.h"
#include"memoryManager.h"
#include"graphChecker.h"

int main(int argc, char** argv) {
    
    if (argc < 2) {
        printf("Usage: %s [inputgraph]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* fileName = argv[1];

    ManWomanGraph* graph = readManWomanGraphInBinary(fileName);
    displaySparse(graph);


    if (!graph) {
        exit(EXIT_FAILURE);
    }

    // call the driver with pointer to hostGraph
    findStableMarriage(graph);

    freeGraph(graph);
 
    //findBipartiteMarriage();
}
