#include <stdio.h>

#include"memoryManager.h"
#include"stdlib.h"

static void releaseMemory(void* ptrToMemory) {
    if (ptrToMemory != NULL) {
        free(ptrToMemory);
        ptrToMemory = NULL;
    }
}

void freeGraph(ManWomanGraph* graph) {
    if (graph) {
        releaseMemory(graph->indicesMan);
        releaseMemory(graph->indicesWoman);
        releaseMemory(graph->mansEvalbyWoman);
        releaseMemory(graph->mansPriorities);
        releaseMemory(graph->womansPriorities);
        releaseMemory(graph);
        printf("Memory released\n");
    }
}

ManWomanGraph* allocateGraph(unsigned int nrMan, unsigned int nrWoman,
        int64_t nrEdges) {

    ManWomanGraph* graph = (ManWomanGraph*) malloc(sizeof (ManWomanGraph));

    graph->nrMan = nrMan;
    graph->nrWoman = nrWoman;
    graph->nrEdges = nrEdges;

    graph->indicesMan = (int64_t*) malloc((nrMan + 1) * sizeof (int64_t));
    graph->indicesWoman = (int64_t*) malloc((nrWoman + 1) * sizeof (int64_t));

    graph->mansPriorities = (unsigned int*) malloc(nrEdges * sizeof (unsigned int));
    graph->mansEvalbyWoman = (unsigned int*) malloc(nrEdges * sizeof (unsigned int));

    graph->womansPriorities = (unsigned int*) malloc(nrEdges * sizeof (unsigned int));

    return graph;

}

CompleteBipartiteGraph* allocateComBipartite(unsigned int nrMan, unsigned int nrWoman) {

    CompleteBipartiteGraph* graph = (CompleteBipartiteGraph*) malloc(sizeof (CompleteBipartiteGraph));

    graph->nrMan = nrMan;
    graph->nrWoman = nrWoman;
    graph->commonRanksOfMen = (unsigned int*) malloc(nrMan * sizeof (unsigned int));
    graph->rankToManId = (unsigned int*) malloc(nrMan * sizeof (unsigned int));

    graph->womenPeferredByMen = (unsigned int*) malloc(nrWoman * sizeof (unsigned int));
    return graph;
}

void freeBipartite(CompleteBipartiteGraph* graph) {

    if (graph) {
        if (graph->womenPeferredByMen) {
            free(graph->womenPeferredByMen);
            graph->womenPeferredByMen = NULL;
        }
        if (graph->rankToManId) {
            free(graph->rankToManId);
            graph->rankToManId = NULL;
        }
        if (graph->commonRanksOfMen) {
            free(graph->commonRanksOfMen);
            graph->commonRanksOfMen = NULL;
        }
        free(graph);
    }
}

