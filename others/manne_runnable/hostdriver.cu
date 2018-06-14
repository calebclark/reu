#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdlib.h>

#include"kernelsdef.h"
#include "hostdriver.h"
#include "datadef.h"
#include "hostDeviceMemoryManager.h"

#include"completeBipartite.h"
#include "memoryManager.h"
#include"bipartiteIO.h"

void printSuitors(ManWomanGraph* hostGraph, unsigned int* suitorsRank) {

    printf("Suitors:");
    for (unsigned int womanId = 0; womanId < min(8, hostGraph->nrWoman); womanId++) {
        unsigned int index = suitorsRank[womanId];
        if (index < hostGraph->nrMan)
            printf("%u ", hostGraph->womansPriorities[ hostGraph->indicesWoman[womanId] + index]);
    }

    printf("\n");
}

bool checkCorrectness(ManWomanGraph* hostGraph, unsigned int* suitorsRank) {

    bool isCorrect = true;
    for (unsigned int womanId = 0; womanId < hostGraph->nrWoman; womanId++) {
        unsigned int index = suitorsRank[womanId];
        if (index < hostGraph->nrMan) {
            unsigned int manId = hostGraph->womansPriorities[ hostGraph->indicesWoman[womanId] + index];
            // check if this man could do any better
            for (unsigned int j = hostGraph->indicesMan[manId]; j < hostGraph->indicesMan[manId + 1]; j++) {

                unsigned int preferredWoman = hostGraph->mansPriorities[j];
                unsigned int rankBypreferredWoman = hostGraph->mansEvalbyWoman[j];


                if (preferredWoman == womanId)
                    break;

                if (suitorsRank[preferredWoman] < rankBypreferredWoman) {

                    //correct path 
                } else {
                    isCorrect = false;
                    printf("\nERROR: In check for correctness; womanId= %u, manId= %u\n", womanId, manId);
                    break;
                }
            }
        }
    }

    if (isCorrect)
        printf("\nCheck for correctness succeeded\n");
    else
        if (isCorrect)printf("\nCheck for correctness failed\n");

    return isCorrect;
}

void checkForDuplicates(ManWomanGraph* hostGraph, unsigned int* suitorsRank) {

    unsigned int* hostSuitors = (unsigned int*) malloc(hostGraph->nrWoman * sizeof (unsigned int));
    memset(hostSuitors, 0, hostGraph->nrWoman * sizeof (unsigned int));

    unsigned int nrCouple = 0;

    for (unsigned int womanId = 0; womanId < hostGraph->nrWoman; womanId++) {
        unsigned int index = suitorsRank[womanId];
        if (index < hostGraph->nrMan) {

            unsigned int manId = hostGraph->womansPriorities[ hostGraph->indicesWoman[womanId] + index];
            unsigned int nrPartner = ++hostSuitors[manId];

            if (nrPartner > 1) {
                printf("\nERROR: A man can't be matched with 2 women;  ManId %u: %u %u !\n", manId, womanId, nrPartner);
                return;
            }
            nrCouple++;
        }
    }

    free(hostSuitors);
    printf("\n#Couple= %u\n", nrCouple);
    double quality = 100.0 * ((double) nrCouple / (double) min(hostGraph->nrMan, hostGraph->nrWoman));
    printf("Quality of matching= %3.3lf", quality);
    printf("\nNo duplicate man!\n");
}


//Build a complete bipartite graph

void makeCompleteBipartieGraph(CompleteBipartiteGraph* graph) {

    for (unsigned int j = 0; j < graph->nrMan; j++) {

        unsigned int rank = (graph->nrMan - 1) - j;
        graph->commonRanksOfMen[j] = rank;
        graph->rankToManId[rank] = j;
    }
    for (unsigned int j = 0; j < graph->nrWoman; j++) {

        graph->womenPeferredByMen[j] = (graph->nrWoman - 1) - j;

    }
}

void computeNrET(unsigned int* ranks, unsigned int nrWoman, unsigned int nrMan) {

    int64_t nrEdgeTraversed = 0;
    for (unsigned int i = 0; i < nrWoman; i++) {
        if (ranks[i] < nrMan)nrEdgeTraversed += (1 + ranks[i]);
    }
    printf("\n#ET = %lu\n", nrEdgeTraversed);
}

void findBipartiteMarriage() {
    assert(cudaSuccess == cudaDeviceReset());
    unsigned int exponent = 1;
    printf("\nEnter exponent: ");
    int retCode = scanf("%u", &exponent);

    assert(retCode > 0 && exponent <= 31);

    //Allocate Graph on host
    CompleteBipartiteGraph* hostgraph = allocateComBipartite(1 << exponent, 1 << exponent);

    // Generate Graph on host
    makeCompleteBipartieGraph(hostgraph);
    //char* bpname= ranking_524288_524288.dat;
    readRanks("ranking_524288_524288.dat", hostgraph);


    //Allocate ptrs for men on device
    unsigned int* devptrsMen = NULL;
    allocateDevMem((void**) &devptrsMen, hostgraph->nrMan * sizeof (unsigned int));

    //Allocate suitor array on device
    unsigned int* devSuitors = NULL;
    allocateDevMem((void**) &devSuitors, hostgraph->nrWoman * sizeof (unsigned int));

    // Allocate Graph on GPU and copy graph from host to GPU
    CompleteBipartiteGraph* devGraph = allocateAndCopyBipartieGraphOnDev(hostgraph);

    //Set suitor of each woman to -1 (initially)
    int fillInValue = (1 << 8) - 1;
    assert(cudaSuccess == cudaMemset(devSuitors, fillInValue, devGraph->nrWoman * sizeof (unsigned int)));

    printf("\n(findBipartiteMarriage) %u %u \n", (*devGraph).nrMan, (*devGraph).nrWoman);


    // lets assign a thread per vertex
    unsigned int nrThreads = 128;

    unsigned int nrBlocks = (devGraph->nrMan + nrThreads - 1) / nrThreads;

    printf("\n#Block = %u #Thread= %u\n", nrBlocks, nrThreads);

    printf("\nGoing to call matchInCompleteBipartite kernel\n");

    cudaEvent_t start, stop;
    assert(cudaSuccess == cudaEventCreate(&start));
    assert(cudaSuccess == cudaEventCreate(&stop));
    assert(cudaSuccess == cudaEventRecord(start, 0));


    matchInCompleteBipartite << <nrBlocks, nrThreads>>>(*devGraph, devptrsMen, devSuitors, devGraph->nrWoman);

    assert(cudaSuccess == cudaEventRecord(stop, 0));
    assert(cudaSuccess == cudaEventSynchronize(stop));

    float elapsedTime = 0.0;
    assert(cudaSuccess == cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\nTime to find match %f milliseconds\n", elapsedTime);

    assert(cudaSuccess == cudaEventDestroy(start));
    assert(cudaSuccess == cudaEventDestroy(stop));

    assert(cudaSuccess == cudaDeviceSynchronize());


    //allocate host memory to retrieve suitors

    unsigned int* suitorsRank = (unsigned int*) malloc(devGraph->nrWoman * sizeof (unsigned int));

    //Copy suitors from GPU to CPU
    copyArray(suitorsRank, devSuitors, devGraph->nrWoman * sizeof (unsigned int), cudaMemcpyDeviceToHost);

    computeNrET(suitorsRank, devGraph->nrWoman, devGraph->nrMan);


    //Go Green!
    freeBipartite(hostgraph);
    freeDevMem(devptrsMen);
    freeDevMem(devSuitors);
    freeCompleteBipartiteDevGraph(devGraph);


}

void findStableMarriage(ManWomanGraph* hostGraph) {

    assert(cudaSuccess == cudaDeviceReset());
    //-----------Allocation--------------------//

    //Allocate graph on device
    ManWomanGraph* devGraph = allocateGraphOnDev(hostGraph);

    //Allocate ptrs for men on device
    unsigned int* devptrsMen = NULL;
    allocateDevMem((void**) &devptrsMen, hostGraph->nrMan * sizeof (unsigned int));

    //Allocate suitor array on device
    unsigned int* devSuitors = NULL;
    allocateDevMem((void**) &devSuitors, hostGraph->nrWoman * sizeof (unsigned int));

    //------------------Initialization---------------//

    //Copy graph from host to device
    copyGraphH2D(hostGraph, devGraph);

    //pointer of each man initially points to start of its woman list
    copyArray(devptrsMen, devGraph->indicesMan, devGraph->nrMan * sizeof (unsigned int), cudaMemcpyDeviceToDevice);

    //Set suitor of each woman to -1 (initially)
    int fillInValue = (1 << 8) - 1;
    assert(cudaSuccess == cudaMemset(devSuitors, fillInValue, devGraph->nrWoman * sizeof (unsigned int)));




    printf("\n(findStableMarriage) %u %u %u\n", (*devGraph).nrMan, (*devGraph).nrWoman, (*devGraph).nrEdges);


    // lets assign a thread per vertex
    unsigned int nrThreads = 128;

    unsigned int nrBlocks = (devGraph->nrMan + nrThreads - 1) / nrThreads;

    printf("\n#Block = %u #Thread= %u\n", nrBlocks, nrThreads);

    printf("\nGoing to call makeMatch kernel\n");


    cudaEvent_t start, stop;
    assert(cudaSuccess == cudaEventCreate(&start));
    assert(cudaSuccess == cudaEventCreate(&stop));
    assert(cudaSuccess == cudaEventRecord(start, 0));


    makeMatch << <nrBlocks, nrThreads>>>(*devGraph, devptrsMen, devSuitors);

    assert(cudaSuccess == cudaEventRecord(stop, 0));
    assert(cudaSuccess == cudaEventSynchronize(stop));

    float elapsedTime = 0.0;
    assert(cudaSuccess == cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\nTime to find match %f milliseconds\n", elapsedTime);

    assert(cudaSuccess == cudaEventDestroy(start));
    assert(cudaSuccess == cudaEventDestroy(stop));

    assert(cudaSuccess == cudaDeviceSynchronize());

    printf("\nReturned from makeMatch kernel\n");

    //allocate host memory to retrieve suitors

    unsigned int* suitorsRank = (unsigned int*) malloc(devGraph->nrWoman * sizeof (unsigned int));

    //Copy suitors from GPU to CPU
    copyArray(suitorsRank, devSuitors, devGraph->nrWoman * sizeof (unsigned int), cudaMemcpyDeviceToHost);

    computeNrET(suitorsRank, devGraph->nrWoman, devGraph->nrMan);

    //printSuitors(hostGraph, suitorsRank);
    checkForDuplicates(hostGraph, suitorsRank);

    checkCorrectness(hostGraph, suitorsRank);
    //Go Green!

    freeManWomanDevGraph(devGraph);
    freeDevMem(devptrsMen);
    freeDevMem(devSuitors);
    free(suitorsRank);

    return;
}
