#include <stdlib.h>
#include <set>
#include <string.h>
#include <assert.h>
#include"unordered_set"
#include"unordered_map"
#include"time.h"
#include "omp.h"
#include"math.h"

#include"dataGenerator.h"

#include"iostream"
using namespace std;

/*
 * Generate Graph with given nrMan and nrWoman; we don't know the nrEdges until
 * we generate
 */
ManWomanGraph* generateGraph_Seq(unsigned int nrMan, unsigned int nrWoman) {


    double start = omp_get_wtime();
    double currTime = omp_get_wtime();
    ManWomanGraph *graph = (ManWomanGraph*) malloc(sizeof (ManWomanGraph));
    graph->nrMan = nrMan;
    graph->nrWoman = nrWoman;
    graph->indicesMan = (int64_t*) malloc((nrMan + 1) * sizeof (int64_t));
    graph->indicesWoman = (int64_t*) malloc((nrWoman + 1) * sizeof (int64_t));

    memset(graph->indicesMan, 0, sizeof (int64_t)*(nrMan + 1));
    memset(graph->indicesWoman, 0, sizeof (int64_t) *(nrWoman + 1));


    srand(time(NULL));

    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to initialize :" << currTime << " seconds " << std::endl;


    //For each man generate a list of woman of length log(nrWoman) to 2*log(nrWoman)
    int maxNrChoice = (int) log2(nrWoman);

    currTime = omp_get_wtime();

    for (int i = 0; i < nrMan; i++) {
        int nrChoice = maxNrChoice + rand() % maxNrChoice; //  maximum  2*log(nrWoman)
        //NOTE:: number of choice for Man i is stored in index (i+1)
        graph->indicesMan[i + 1] = nrChoice;
    }

    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to generate #choices (sequentially):" << currTime << " seconds " << std::endl;



#ifdef DISPLAY_DETAILS
    //Print #choice per man
    for (int i = 0; i <= nrMan; i++) {
        std::cout << graph->indicesMan[i] << " ";
    }
    std::cout << std::endl;

#endif

    //prefix sum
    for (int i = 0; i < nrMan; i++) {
        graph->indicesMan[i + 1] += graph->indicesMan[i];
    }

#ifdef DISPLAY_DETAILS
    //after prefix sum
    for (int i = 0; i <= nrMan; i++) {
        std::cout << graph->indicesMan[i] << " ";
    }
#endif


    int64_t szEdgelist = graph->indicesMan[nrMan];

    std::cout << std::endl << "Generating a graph with " << nrMan << " men and " << nrWoman << " women" << std::endl;

    std::cout << "size of Edge list: " << szEdgelist << " * 2" << std::endl;

    graph->mansPriorities = (unsigned int*) malloc(szEdgelist * sizeof (unsigned int));


    std::cout << "Start generating choices for each man" << std::endl;

    //For each man generate preferred women


    currTime = omp_get_wtime();



    for (int i = 0; i < nrMan; i++) {

        std::unordered_set<int> myset;
        std::unordered_set<int>::iterator it;

        int nrElement = graph->indicesMan[i + 1] - graph->indicesMan[i ];

        while (myset.size() < nrElement) {
            int womanId = rand() % nrWoman;
            myset.insert(womanId);
            //if (i == 0)std::cout << womanId << "-";
        }

        //if (i == 0)std::cout << std::endl << nrElement << "::" << myset.size() << std::endl;

        if (nrElement != myset.size())
            std::cout << "ERROR: Don't have enough choices for Man " << i << std::endl;

        it = myset.begin();

        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            graph->mansPriorities[j] = *it;
            it++;
        }

        myset.clear();
    }



    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to generate random choices (sequentially):" << currTime << " seconds " << std::endl;



#ifdef DISPLAY_DETAILS
    //print
    std::cout << std::endl;
    for (int i = 0; i < nrMan; i++) {
        std::cout << i << " : ";
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            std::cout << graph->mansPriorities[j] << " ";
        }
        std::cout << std::endl;
    }
#endif


    std::cout << "Count how many times each woman appeared" << std::endl;

#ifdef DISPLAY_DETAILS
    //print
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i <= nrWoman; i++) {
        std::cout << graph->indicesWoman[i] << " ";
    }
    std::cout << std::endl;
#endif


    // for each woman count how many times they appeared in the lists of men

    currTime = omp_get_wtime();
    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            int womanId = graph->mansPriorities[j];
            //__sync_fetch_and_add(&graph->indicesWoman[womanId + 1], 1);
            graph->indicesWoman[womanId + 1]++;
        }
    }
    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to count women:" << currTime << " seconds " << std::endl;


#ifdef DISPLAY_DETAILS
    //print
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i <= nrWoman; i++) {
        std::cout << graph->indicesWoman[i] << " ";
    }
    std::cout << std::endl;
#endif


    //prefix sum
    for (int i = 0; i < nrWoman; i++) {
        graph->indicesWoman[i + 1] += graph->indicesWoman[i];
    }


#ifdef DISPLAY_DETAILS
    //print
    std::cout << std::endl;
    for (int i = 0; i <= nrWoman; i++) {
        std::cout << graph->indicesWoman[i] << " ";
    }
    std::cout << std::endl;
#endif

    int64_t sumWomanfreq = graph->indicesWoman[nrWoman];

    assert(szEdgelist == sumWomanfreq);

    graph->nrEdges = szEdgelist;

    graph->womansPriorities = (unsigned int*) malloc(sumWomanfreq * sizeof (unsigned int));



    int64_t *womanPtrs = (int64_t*) malloc(sizeof (int64_t)*(nrWoman + 1));

    memcpy(womanPtrs, graph->indicesWoman, (nrWoman + 1) * sizeof (int64_t));

#ifdef DISPLAY_DETAILS

    //print
    std::cout << "temporaryPtrs:" << std::endl;
    for (int i = 0; i <= nrWoman; i++) {
        std::cout << womanPtrs[i] << " ";
    }

#endif


    std::cout << "Copy corresponding men into list of each woman" << std::endl;

    currTime = omp_get_wtime();
    //copy list of men into each woman lists

    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {

            int womanId = graph->mansPriorities[j];

            // go to the list of particular woman and put the man there
            int64_t position = womanPtrs[womanId]++; //-----------------------------------<
            //int position = __sync_fetch_and_add(&womanPtrs[womanId], 1);
            graph->womansPriorities[position] = i; // i is the manId

        }
    }
    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to copy  (sequentially):" << currTime << " seconds " << std::endl;



#ifdef DISPLAY_DETAILS

    //print
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < nrWoman; i++) {
        std::cout << i << " : ";
        for (int64_t j = graph->indicesWoman[i]; j < graph->indicesWoman[i + 1]; j++) {
            std::cout << graph->womansPriorities[j] << " ";
        }
        std::cout << std::endl;
    }
#endif


    // shuffle men in the list each woman

    //
    //TO DO
    //


    std::cout << "Find ranks of each man in women lists" << std::endl;


    currTime = omp_get_wtime();

    //Construct mans evaluation by woman
    unordered_map<int, int>* evalMap = new unordered_map<int, int>[nrMan];

    for (int i = 0; i < nrWoman; i++) {
        int rankOffset = graph->indicesWoman[i];
        for (int64_t j = graph->indicesWoman[i]; j < graph->indicesWoman[i + 1]; j++) {

            int manId = graph->womansPriorities[j];
            // multiple threads responsible for women may try to update same map for "Man"

            evalMap[manId][i] = j - rankOffset;
        }
    }


    currTime = omp_get_wtime() - currTime;

    std::cout << "Time to find ranks (sequentially):" << currTime << " seconds " << std::endl;





    std::cout << "Copy ranks of man into corresponding array" << std::endl;

    graph->mansEvalbyWoman = (unsigned int*) malloc(szEdgelist * sizeof (unsigned int));


    currTime = omp_get_wtime();

    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            int womanId = graph->mansPriorities[j];
            graph->mansEvalbyWoman[j] = evalMap[i][womanId];
        }
    }

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to copy ranks (sequentially):" << currTime << " seconds " << std::endl;

    currTime = omp_get_wtime();

    for (int i = 0; i < nrMan; i++) {
        evalMap[i].clear();
    }

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to clear maps:" << currTime << " seconds " << std::endl;



    currTime = omp_get_wtime();
    //Free temporary buffers
    free(womanPtrs);

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to free an index array:" << currTime << " seconds " << std::endl;


    currTime = omp_get_wtime();

    delete [] evalMap;

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to free maps:" << currTime << " seconds " << std::endl;


    start = omp_get_wtime() - start;
    std::cout << "Total: " << start << std::endl;

    return graph;

}


