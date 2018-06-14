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
ManWomanGraph* generateGraph(unsigned int nrMan, unsigned int nrWoman, unsigned int nrThreads) {

    double start = omp_get_wtime();

    double currTime = omp_get_wtime();

    ManWomanGraph *graph = (ManWomanGraph*) malloc(sizeof (ManWomanGraph));
    graph->nrMan = nrMan;
    graph->nrWoman = nrWoman;
    graph->indicesMan = (int64_t*) malloc((nrMan + 1) * sizeof (int64_t));
    graph->indicesWoman = (int64_t*) malloc((nrWoman + 1) * sizeof (int64_t));

    memset(graph->indicesMan, 0, sizeof (int64_t)*(nrMan + 1));
    memset(graph->indicesWoman, 0, sizeof (int64_t) *(nrWoman + 1));

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to initialize:" << currTime << " seconds " << std::endl;

    //srand(time(NULL));
    //omp_set_dynamic(0); // Explicitly disable dynamic teams
    //omp_set_num_threads(4);


#ifdef DISPLAY_DETAILS
    std::cout << "Before Parallel Construct: " << "|P| = " << omp_get_num_procs() << "|T| = " << omp_get_num_threads() << std::endl;
#pragma omp parallel
    {
#pragma omp single
        cout << "threads=" << omp_get_num_threads() << endl;
    }
#endif

    currTime = omp_get_wtime();

    //For each man generate a list of woman of length log(nrWoman) to 2*log(nrWoman)
    int maxNrChoice = (int) log2(nrWoman);

#pragma omp parallel num_threads(nrThreads)
    {
        srand(int(time(NULL)) ^ omp_get_thread_num());

#pragma omp for
        for (int i = 0; i < nrMan; i++) {
            int nrChoice = maxNrChoice + rand() % maxNrChoice; //  maximum  2*log(nrWoman)
            //NOTE:: number of choice for Man i is stored in index (i+1)
            graph->indicesMan[i + 1] = nrChoice;

            if (omp_get_thread_num() == 0 && i == 0) {
                std::cout << "*|P| = " << omp_get_num_procs() << "  |T| = " << omp_get_num_threads() << " i:" << i << std::endl;
            }
        }
    }
    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to decide #choices:" << currTime << " seconds " << std::endl;



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

    std::cout << "size of Edge list = " << szEdgelist << " * 2" << std::endl;

    graph->mansPriorities = (unsigned int*) malloc(szEdgelist * sizeof (unsigned int));


    std::cout << "Start generating choices for each man" << std::endl;

    currTime = omp_get_wtime();

    //For each man generate preferred women
#pragma omp parallel num_threads(nrThreads)
    {
        srand(int(time(NULL)) ^ omp_get_thread_num());
#pragma omp for
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

            if (omp_get_thread_num() == 0 && i == 0) {
                std::cout << "**|P| = " << omp_get_num_procs() << "  |T| = " << omp_get_num_threads() << " i:" << i << std::endl;
            }

        }
    }


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to generate choices:" << currTime << " seconds " << std::endl;



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

    currTime = omp_get_wtime();
    // for each woman count how many times they appeared in the lists of men
#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            int womanId = graph->mansPriorities[j];
            __sync_fetch_and_add(&graph->indicesWoman[womanId + 1], 1);
            //graph->indicesWoman[womanId + 1]++;
        }
    }


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to count each woman:" << currTime << " seconds " << std::endl;

#ifdef DISPLAY_DETAILS
    //print
    std::cout << std::endl;
    std::cout << std::endl;
    for (int i = 0; i <= nrWoman; i++) {
        std::cout << graph->indicesWoman[i] << " ";
    }
    std::cout << std::endl;
#endif

    currTime = omp_get_wtime();

    //prefix sum
    for (int i = 0; i < nrWoman; i++) {
        graph->indicesWoman[i + 1] += graph->indicesWoman[i];
    }

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to compute prefix sum :" << currTime << " seconds " << std::endl;


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
#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {

            int womanId = graph->mansPriorities[j];

            // go to the list of particular woman and put the man there
            int64_t position = __sync_fetch_and_add(&womanPtrs[womanId], 1);
            graph->womansPriorities[position] = i; // i is the manId

        }
    }


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to build list(of men) for women:" << currTime << " seconds " << std::endl;

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

    omp_lock_t *locksForMan = (omp_lock_t*) malloc(nrMan * sizeof (omp_lock_t));

#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        omp_init_lock(&locksForMan[i]);
    }


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to create locks:" << currTime << " seconds " << std::endl;

    std::cout << "Created lock for each man" << std::endl;

    currTime = omp_get_wtime();

    //Construct mans evaluation by woman
    unordered_map<int, int>* evalMap = new unordered_map<int, int>[nrMan];



#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrWoman; i++) {
        int64_t rankOffset = graph->indicesWoman[i];
        for (int64_t j = graph->indicesWoman[i]; j < graph->indicesWoman[i + 1]; j++) {

            int manId = graph->womansPriorities[j];
            // multiple threads responsible for women may try to update same map for "Man"

            omp_set_lock(&locksForMan[manId]);

            evalMap[manId][i] = j - rankOffset;

            omp_unset_lock(&locksForMan[manId]);

        }
    }

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to find ranks:" << currTime << " seconds " << std::endl;



    currTime = omp_get_wtime();

#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        omp_destroy_lock(&locksForMan[i]);
    }
    //Free memory occupied by locks
    free(locksForMan);

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to destroy locks:" << currTime << " seconds " << std::endl;



    std::cout << "Copy ranks of man into corresponding array" << std::endl;

    graph->mansEvalbyWoman = (unsigned int*) malloc(szEdgelist * sizeof (unsigned int));


    currTime = omp_get_wtime();

#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        for (int64_t j = graph->indicesMan[i]; j < graph->indicesMan[i + 1]; j++) {
            int womanId = graph->mansPriorities[j];
            graph->mansEvalbyWoman[j] = evalMap[i][womanId];
        }
    }


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to copy ranks:" << currTime << " seconds " << std::endl;


    currTime = omp_get_wtime();

#pragma omp parallel for num_threads(nrThreads)
    for (int i = 0; i < nrMan; i++) {
        evalMap[i].clear();
    }

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to clear maps:" << currTime << " seconds " << std::endl;


    currTime = omp_get_wtime();
    //Free temporary buffers
    free(womanPtrs);


    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to free:" << currTime << " seconds " << std::endl;


    currTime = omp_get_wtime();
    delete [] evalMap;

    currTime = omp_get_wtime() - currTime;
    std::cout << "Time to delete maps:" << currTime << " seconds " << std::endl;



    start = omp_get_wtime() - start;
    std::cout << "Total:" << start << " seconds " << std::endl;

    return graph;

}

