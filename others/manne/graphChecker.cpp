
#include <stdio.h>

#include"stdlib.h"
#include"graphChecker.h"

void displayGraph(ManWomanGraph* graph) {

    printf("\nFrom Man Side:\n");
    for (int i = 0; i < graph->nrMan; i++) {
        printf("%d : ", i);
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {

            int womanId = graph->mansPriorities[j];
            int priorityByWoman = graph->mansEvalbyWoman[j];
            printf("%d(%d) ", womanId, priorityByWoman);
        }
        printf("\n");
    }
}

bool checkManWomanGraph(ManWomanGraph* graph, int nrThreads) {

    bool returnValue = true;
#pragma omp parallel for num_threads(nrThreads)  reduction(&& : returnValue)
    for (int i = 0; i < graph->nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {

            int womanId = graph->mansPriorities[j];
            int priorityByWoman = graph->mansEvalbyWoman[j];
            //check if man i is found there
            if (graph->womansPriorities[ graph->indicesWoman[womanId] + priorityByWoman] != i) {
                printf("****ERROR*****: woman is missing; manId = %d, womanId = %d\n", i, womanId);
                returnValue = returnValue && false;
            }
        }
    }


    return returnValue;
}

void displaySparse(ManWomanGraph* graph) {

    unsigned int i = graph->nrMan - 1;
    printf("%d : ", i);
    for (int j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {

        int womanId = graph->mansPriorities[j];
        int priorityByWoman = graph->mansEvalbyWoman[j];
        printf("%d(%d) ", womanId, priorityByWoman);
    }
    printf("\n");

}

