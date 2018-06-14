#include <omp.h>

#include"dataGenerator.h"
#include"graphChecker.h"
#include"graphIO.h"
#include"memoryManager.h"

#include"hostdriver.h"
#include"iostream"
#include"time.h"
using namespace std;

#define NUM_THREAD 36
#define NR_MAN  20000000

static void callSeqGenerator() {




    double currTime = omp_get_wtime();

    //Generate a graph
    ManWomanGraph* graph = generateGraph_Seq(NR_MAN, NR_MAN);

    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to generate (sequentially):" << currTime << " seconds " << std::endl;

    //displaySparse(graph);

    //displayGraph(graph);

    //Check if it's a legal stable marriage graph
    if (checkManWomanGraph(graph, NUM_THREAD)) {
        std::cout << "Valid Graph for stable marriage" << std::endl;
    } else {
        std::cout << "Error: Invalid Graph Format" << std::endl;
    }

    freeGraph(graph);
}

int main(int argc, char**argv) {
    /*
    for (int i = 0; i < argc; i++) {
        std::cout << argv[i] << std::endl;
    }
     */

    //callSeqGenerator();

    double currTime = omp_get_wtime();

    unsigned int N;
    if(argc > 1){
        N = atoi(argv[1]);
    } else {
        printf("No N given\n");
        exit(1);
    }

    //Generate a graph
    ManWomanGraph* graph = generateGraph(N, N, NUM_THREAD);

    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to generate:" << currTime << " seconds " << std::endl;

    //displaySparse(graph);

    //displayGraph(graph);

    //Check if it's a legal stable marriage graph
    if (checkManWomanGraph(graph, NUM_THREAD)) {
        std::cout << "Valid Graph for stable marriage" << std::endl;
    } else {
        std::cout << "Error: Invalid Graph Format" << std::endl;
    }


    currTime = omp_get_wtime();
    //write it in a file
    char* fileName = writeManWomanGraphInBinary(graph);

    std::cout << "File:" << fileName << std::endl;
    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to write:" << currTime << " seconds " << std::endl;

    //Free the graph
    freeGraph(graph);

    //Read input graph from file


    std::cout << "Going to read from " << fileName << std::endl;
    currTime = omp_get_wtime();

    ManWomanGraph* inputGraph = readManWomanGraphInBinary(fileName);

    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to read:" << currTime << " seconds " << std::endl;

    //displaySparse(inputGraph);
    //display it
    //displayGraph(inputGraph);

    //free it
    freeGraph(inputGraph);

    free(fileName);
    // call driver to find stable Marriage
    //findStableMarriage();
    return 0;
}
