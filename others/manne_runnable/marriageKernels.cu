#include <stdio.h>

#include"kernelsdef.h"


#ifdef RUNONGPU

__device__
#endif
void printGraphFromGPU(ManWomanGraph *graph) {
    for (unsigned int i = 0; i < graph->nrMan; i++) {
        printf("%u: ", i);
        for (unsigned int j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            printf("%u(%u) ", graph->mansPriorities[j], graph->mansEvalbyWoman[j]);
        }
        printf("\n");
    }
}


#ifdef RUNONGPU

__device__
#endif
void findWoman(ManWomanGraph graph, unsigned int *currPtrsMen,
        unsigned int *suitorsRanks, unsigned int manId) {


    unsigned int rankPrev;
    unsigned int prevMan = graph.nrMan;

    bool done = false;

    while (!done) {
        unsigned int currPos;
        //start from the position of current pointer; search onwards
        for (currPos = currPtrsMen[manId]; currPos < graph.indicesMan[manId + 1]; currPos++) {

            unsigned int womanId = graph.mansPriorities[currPos];
            unsigned int rankOfMan = graph.mansEvalbyWoman[currPos]; // according the woman

            //rank of current offer for the woman
            unsigned int rankOfCurrOffer = suitorsRanks[womanId]; // according the woman

            // Man with  smaller rank is preferred by women

            if (rankOfMan > rankOfCurrOffer)
                continue;

            if (rankOfMan == rankOfCurrOffer)
                printf("Can't Happen %u \n", manId);


            //unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);

            rankPrev = atomicCAS(&suitorsRanks[womanId], rankOfCurrOffer, rankOfMan);

            if (rankPrev == rankOfCurrOffer) { // i.e. manId has become current suitor

                //Find the previous man who has been dislodged 

                if (rankPrev < graph.nrMan) { //There was a "valid" previous suitor

                    prevMan = graph.womansPriorities[ graph.indicesWoman[womanId] + rankPrev];

                    done = false;

                } else { // There was no "valid" previous suitor

                    done = true;
                }

                break;

            } else { //someone has changed suitorsRanks[womanId] meanwhile

                currPos--; // let try again
            }
        }

        //save the pointer
        currPtrsMen[manId] = currPos + 1;

        //done == false but  has reached end of list
        if (done == false && currPos >= graph.indicesMan[manId + 1])
            done = true;


        if (done == false) {
            unsigned int oldManId = manId;
            manId = prevMan;
            //printf("%u->%u,", oldManId, manId);
        }
    }
}

#ifdef RUNONGPU

__global__
#endif
void makeMatch(ManWomanGraph graph, unsigned int *currPtrsMen, unsigned int *suitorsRanks) {

    unsigned int blkId = blockIdx.x;
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    /*
        if (!threadId) {

            printf("\n(makeMatch) %u %u %u %p %p %p %p %p\n", graph.nrMan, graph.nrWoman,
                    graph.nrEdges, graph.indicesMan, graph.indicesWoman,
                    graph.mansPriorities, graph.mansEvalbyWoman, graph.womansPriorities);
        }

        if (graph.nrMan <= 8 && graph.nrWoman <= 8) {
            if (!threadId) {
                printGraphFromGPU(&graph);
            }
        }

        if (!threadId) {
            unsigned int i = graph.nrMan - 1;

            printf("Man_%d: ", i);

            for (int j = graph.indicesMan[i]; j < graph.indicesMan[i + 1]; j++) {
                unsigned int womanId = graph.mansPriorities[j];
                unsigned int eval = graph.mansEvalbyWoman[j];
                printf("%d(%d)-", womanId, eval);
            }

            printf("\nSuitors:");

            for (int i = 0; i < graph.nrWoman; i++) {
                printf("%u ", suitorsRanks[i]);

            }

            printf("\nIndicesMen: ");

            for (int i = 0; i <= graph.nrMan; i++) {
                printf("%u ", graph.indicesMan[i]);
            }

            printf("\nCurrPtrsMen: ");

            for (int i = 0; i < graph.nrMan; i++) {
                printf("%u ", currPtrsMen[i]);
            }
            printf("\n");

        }

        if (!threadIdx.x)
            printf("\nbId=%d tId= %d\n", blkId, threadId);
     */
    if (threadId < graph.nrMan) {
        findWoman(graph, currPtrsMen, suitorsRanks, threadId);
    }

    return;
}
