#include"kernelsdef.h"
#include"stdio.h"
#ifdef RUNONGPU

__device__
#endif
void findWoman_CompleteBipartite(CompleteBipartiteGraph graph, unsigned int *currPtrsMen,
        unsigned int *suitorsRanks, unsigned int szNbrListForMen, unsigned int manId) {

    unsigned int rankPrev;
    unsigned int prevMan = graph.nrMan;

    bool done = false;

    while (!done) {

        if(manId == graph.nrMan-1)
            printf("\nInside matchInCompleteBipartite\n");
        unsigned int currPos;
        //start from the position of current pointer; search onwards
        for (currPos = currPtrsMen[manId]; currPos < szNbrListForMen; currPos++) {

            unsigned int womanId = graph.womenPeferredByMen[currPos];
            unsigned int rankOfMan = graph.commonRanksOfMen[manId]; // according the woman; indeed by all woman

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

                if (rankPrev < szNbrListForMen) { //There was a "valid" previous suitor

                    prevMan = graph.rankToManId[rankPrev];

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
        if (done == false && currPos >= szNbrListForMen)
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
void matchInCompleteBipartite(CompleteBipartiteGraph graph,
        unsigned int *currPtrsMen, unsigned int *suitorsRanks,
        unsigned int szNbrListForMen) {

    unsigned int blkId = blockIdx.x;
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadId==0)
        printf("\nInside matchInCompleteBipartite\n");
    if (threadId < graph.nrMan) {

        findWoman_CompleteBipartite(graph, currPtrsMen, suitorsRanks, szNbrListForMen, threadId);

    }
}


