
#include <stdio.h>

#include "stddef.h"
#include "assert.h"
#include "hostDeviceMemoryManager.h"
#include"cuda.h"
#include"cuda_runtime.h"

void allocateDevMem(void** memPtr, size_t nrBytes) {

    assert(cudaSuccess == cudaMalloc(memPtr, nrBytes));
}

void freeDevMem(void* memPtr) {

    //printf("\n memPtr = %p\n", memPtr);
    //printf("\nCUDA FREE\n");
    if (memPtr != NULL)
        assert(cudaSuccess == cudaFree(memPtr));
}

/*
 * Caller Module needs to verify the sizes of corresponding array
 */
void copyArray(void* dest, void *source, size_t nrBytes, cudaMemcpyKind direction) {

    assert(dest != NULL && source != NULL && nrBytes > 0);

    assert(cudaSuccess == cudaMemcpy(dest, source, nrBytes, direction));
}

ManWomanGraph* allocateGraphOnDev(ManWomanGraph* graph) {
    ManWomanGraph* devGraph = NULL;

    if (graph) {

        // Only member pointers are allocated on device
        devGraph = (ManWomanGraph*) malloc(sizeof (ManWomanGraph));


        if (graph->indicesMan != NULL && graph->nrMan > 0) {

            // printf("\ndevGraph->indicesMan= %p\n", devGraph->indicesMan);

            devGraph->nrMan = graph->nrMan;

            allocateDevMem((void**) &devGraph->indicesMan, (graph->nrMan + 1) * sizeof (unsigned int));

            //printf("\ndevGraph->indicesMan= %p\n", devGraph->indicesMan);

        } else {
            printf("\nERROR: indicesMan not allocated on device\n");
        }

        if (graph->indicesWoman != NULL && graph->nrWoman > 0) {
            devGraph->nrWoman = graph->nrWoman;
            allocateDevMem((void**) &devGraph->indicesWoman, (graph->nrWoman + 1) * sizeof (unsigned int));
        } else {
            printf("\nERROR: indicesWoman not allocated on device\n");
        }

        if (graph->mansPriorities != NULL && graph->mansEvalbyWoman != NULL &&
                graph->womansPriorities != NULL && graph->nrEdges > 0) {

            devGraph->nrEdges = graph->nrEdges;

            size_t nrBytes2Allocate = graph->nrEdges * sizeof (unsigned int);

            allocateDevMem((void**) &devGraph->mansPriorities, nrBytes2Allocate);
            allocateDevMem((void**) &devGraph->mansEvalbyWoman, nrBytes2Allocate);
            allocateDevMem((void**) &devGraph->womansPriorities, nrBytes2Allocate);
        } else {
            printf("\nmansPriorities or mansEvalbyWoman or womansPriorities not allocated on device\n");
        }

    }
    return devGraph;
}

void copyGraphH2D(ManWomanGraph* hostGraph, ManWomanGraph* devGraph) {

    if (hostGraph != NULL && devGraph != NULL) {

        //Make sure pointers are not NULL
        assert(hostGraph->indicesMan != NULL && devGraph->indicesMan != NULL &&
                devGraph->nrMan >= hostGraph->nrMan && hostGraph->nrMan > 0);

        assert(cudaSuccess == cudaMemcpy(devGraph->indicesMan, hostGraph->indicesMan,
                (hostGraph->nrMan + 1) * sizeof (unsigned int), cudaMemcpyHostToDevice));

        assert(hostGraph->indicesWoman != NULL && devGraph->indicesWoman != NULL
                && devGraph->nrWoman >= hostGraph->nrWoman && hostGraph->nrWoman > 0);

        assert(cudaSuccess == cudaMemcpy(devGraph->indicesWoman, hostGraph->indicesWoman,
                (hostGraph->nrWoman + 1) * sizeof (unsigned int), cudaMemcpyHostToDevice));

        assert(devGraph->nrEdges >= hostGraph->nrEdges && hostGraph->nrEdges > 0);


        size_t nrBytes2Cpy = hostGraph->nrEdges * sizeof (unsigned int);


        assert(cudaSuccess == cudaMemcpy(devGraph->mansPriorities,
                hostGraph->mansPriorities, nrBytes2Cpy, cudaMemcpyHostToDevice));

        assert(cudaSuccess == cudaMemcpy(devGraph->mansEvalbyWoman,
                hostGraph->mansEvalbyWoman, nrBytes2Cpy, cudaMemcpyHostToDevice));

        assert(cudaSuccess == cudaMemcpy(devGraph->womansPriorities,
                hostGraph->womansPriorities, nrBytes2Cpy, cudaMemcpyHostToDevice));



    } else {
        printf("\nBoth host and device graphs need to allocated beforehand\n");
    }
}

void freeManWomanDevGraph(ManWomanGraph* devGraph) {

    if (devGraph) {
        freeDevMem(devGraph->indicesMan);
        freeDevMem(devGraph->indicesWoman);
        freeDevMem(devGraph->mansPriorities);
        freeDevMem(devGraph->mansEvalbyWoman);
        freeDevMem(devGraph->womansPriorities);
        free(devGraph);
        printf("\nFreed ManWomanGraph\n");
    }

}

void freeCompleteBipartiteDevGraph(CompleteBipartiteGraph* devGraph) {
    if (devGraph) {
        freeDevMem(devGraph->commonRanksOfMen);
        freeDevMem(devGraph->rankToManId);
        freeDevMem(devGraph->womenPeferredByMen);
        free(devGraph);
        printf("\nFreed CompleteBipartiteGraph\n");
    }
}

CompleteBipartiteGraph* allocateAndCopyBipartieGraphOnDev(CompleteBipartiteGraph* graph) {

    CompleteBipartiteGraph* devGraph = (CompleteBipartiteGraph*) malloc(sizeof (CompleteBipartiteGraph));

    devGraph->nrMan = graph->nrMan;
    devGraph->nrWoman = graph->nrWoman;

    assert(graph != NULL && graph->nrMan > 0 && graph->nrWoman > 0);


    assert(cudaSuccess == cudaMalloc((void**) &devGraph->womenPeferredByMen,
            graph->nrWoman * sizeof (unsigned int)));

    assert(cudaSuccess == cudaMemcpy(devGraph->womenPeferredByMen, graph->womenPeferredByMen,
            graph->nrWoman * sizeof (unsigned int), cudaMemcpyHostToDevice));



    assert(cudaSuccess == cudaMalloc((void**) &devGraph->commonRanksOfMen,
            graph->nrMan * sizeof (unsigned int)));

    assert(cudaSuccess == cudaMemcpy(devGraph->commonRanksOfMen, graph->commonRanksOfMen,
            graph->nrMan * sizeof (unsigned int), cudaMemcpyHostToDevice));


    assert(cudaSuccess == cudaMalloc((void**) &devGraph->rankToManId,
            graph->nrMan * sizeof (unsigned int)));

    assert(cudaSuccess == cudaMemcpy(devGraph->rankToManId, graph->rankToManId,
            graph->nrMan * sizeof (unsigned int), cudaMemcpyHostToDevice));

    return devGraph;

}
