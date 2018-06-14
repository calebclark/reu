#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>    /* for getopt */
#include <assert.h>
#include <stdint.h>
#include "graphIO.h"
#include "bipartiteIO.h"
#include "memoryManager.h"
#include "threadsDefinition.h"

int compare_doubles(const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

inline unsigned int get_man_priority(CompleteBipartiteGraph *graph, unsigned int male_id, int rank) {
    return graph->womenPeferredByMen[rank];
}

inline unsigned int get_womans_evaluation(CompleteBipartiteGraph *graph, unsigned int male_id, int rank) {
    return graph->commonRanksOfMen[male_id];
}
inline unsigned int get_woman_priority(CompleteBipartiteGraph *graph, unsigned int woman_id, int rank) {
    return graph->rankToManId[rank];
}

inline unsigned int get_num_ranks(CompleteBipartiteGraph *graph, unsigned int male_id) {
    return graph->nrWoman;
}

unsigned int calculate_match_count(CompleteBipartiteGraph *graph, int *manCurrentRanking) {
    int match_count = 0;
    #pragma omp parallel for reduction(+:match_count)
    for(unsigned int i = 0; i < graph->nrMan; i++) {
        if(manCurrentRanking[i] != INT_MAX && manCurrentRanking[i] != -1){
            match_count += 1;
        } else {
            printf("%d does not have a match\n", i);
        }
    }
    return match_count;
}


size_t calculate_edges_traversed(CompleteBipartiteGraph *graph, int *manCurrentRanking) {
    size_t edges_traversed = 0;
    #pragma omp parallel for reduction(+:edges_traversed)
    for(unsigned int i = 0; i < graph->nrMan; i++) {
        if(manCurrentRanking[i] == INT_MAX) {
            edges_traversed += graph->nrWoman;
        } else {
            assert(manCurrentRanking[i] != -1);
            edges_traversed += (manCurrentRanking[i] + 1);
        }
    }
    return edges_traversed;
}

unsigned int find_next_match(CompleteBipartiteGraph *graph, omp_lock_t *w_locks, int *manCurrentRanking, int *womanCurrentRanking, unsigned int male_id) {
    bool done = false;

    int num_ranks = graph->nrWoman;
    while(!done) {
        int man_current_rank = ++manCurrentRanking[male_id];
        if(man_current_rank < num_ranks){
            unsigned int woman_id = get_man_priority(graph, male_id, man_current_rank); //graph->mansPriorities[man_index + man_current_rank];
            unsigned int man_ranked_as = get_womans_evaluation(graph, male_id, man_current_rank); //graph->mansEvalbyWoman[man_index + man_current_rank];

            // CAS-based:
            while(true) {
                unsigned int womans_current_rank = womanCurrentRanking[woman_id];
                if(man_ranked_as < womans_current_rank) {
                    if(__sync_bool_compare_and_swap(&womanCurrentRanking[woman_id], womans_current_rank, man_ranked_as)){
                        if(womans_current_rank != INT_MAX)
                            male_id = get_woman_priority(graph, woman_id, womans_current_rank); //graph->womansPriorities[graph->indicesWoman[woman_id] + womans_current_rank];
                        else
                            male_id = -1;
                        done = true;
                        break;
                    }

                } else {
                    break;
                }
            }
            /*
            // lock based:
            int *womans_current_rank = &womanCurrentRanking[woman_id];
            if(man_ranked_as < *womans_current_rank) {
                omp_set_lock(&w_locks[woman_id]);
                if(man_ranked_as < *womans_current_rank) {
                    if(*womans_current_rank != INT_MAX)
                        male_id = get_woman_priority(graph, woman_id, *womans_current_rank);
                    else
                        male_id = -1;
                    *womans_current_rank = man_ranked_as;
                    done = true;

                }
                omp_unset_lock(&w_locks[woman_id]);

            }
            */
        } else {
            manCurrentRanking[male_id] = INT_MAX;
            male_id = -1;
            done = true;
        }
    }
    return male_id;
}

double experiment_stack(CompleteBipartiteGraph *graph, omp_lock_t *w_locks, int *manCurrentRanking, int *womanCurrentRanking, bool debug) {

    double start = omp_get_wtime();


    #pragma omp parallel for
    for(size_t i = 0; i < graph->nrMan; i++) {
        int male_id = i;
        // Check that we still have a male to process and if we have more women to consider.
        while(male_id != -1) {
            male_id = find_next_match(graph, w_locks, manCurrentRanking, womanCurrentRanking, male_id);
        }


    }
    double end = omp_get_wtime();
    double delta = end - start;


    return delta;
}

double experiment_queue(CompleteBipartiteGraph *graph, omp_lock_t *w_locks, int *manCurrentRanking, int *womanCurrentRanking, bool debug) {

    double start = omp_get_wtime();


    // Pass 1
    #pragma omp parallel
    {

        int error;
        unsigned int *queue;
        size_t queue_len = graph->nrMan + 10;

        if((error = posix_memalign((void**)&queue, 64, sizeof(unsigned int)*queue_len )) != 0){
            printf("Could not allocate queue: %d\n", error);
            exit(error);
        }

        long in = 0;
        long out = 0;

        // Stage 1
        #pragma omp for
        for(size_t i = 0; i < graph->nrMan; i++) {
            queue[in++] = i;
        }

        while(in != out) {
            int male_id = queue[out++];
            out %= queue_len;

            male_id = find_next_match(graph, w_locks, manCurrentRanking, womanCurrentRanking, male_id);
            if(male_id != -1) {
                queue[in++] = male_id;
                in %= queue_len;
            }
        }

        free(queue);

    }
    double end = omp_get_wtime();
    double delta = end - start;
    return delta;
}

bool run_correctness_check(CompleteBipartiteGraph *graph, omp_lock_t *w_locks, double (*function)(CompleteBipartiteGraph *, omp_lock_t *, int *, int *, bool)) {

    int error;
    int *manCurrentRanking;
    int *womanCurrentRanking;

    if((error = posix_memalign((void **)&manCurrentRanking, 64, sizeof(int)*graph->nrMan)) != 0){
        printf("Could not allocate manCurrentRanking: %d\n", error);
        exit(error);
    }
    if((error = posix_memalign((void **)&womanCurrentRanking, 64, sizeof(int)*graph->nrWoman)) != 0){
        printf("Could not allocate womanCurrentRanking: %d\n", error);
        exit(error);
    }

    printf("allocated space for %d men\n", graph->nrMan);

    for(size_t i = 0; i < graph->nrMan; i++) {
        manCurrentRanking[i] = -1;
    }
    for(size_t i = 0; i < graph->nrWoman; i++) {
        womanCurrentRanking[i] = INT_MAX;
    }

    printf("Running first experiment\n");
    (*function)(graph, w_locks,manCurrentRanking, womanCurrentRanking, true);

    bool correct = true;

        printf("checking if stable: ");
        #pragma omp parallel for reduction(&&:correct)
        for(size_t m = 0; m < graph->nrMan; m++) {

            unsigned int mans_current_rank = manCurrentRanking[m];
            for(size_t r = 0; r < graph->nrWoman && correct; r++) {


                unsigned int f = get_man_priority(graph, m, r);
                unsigned int man_ranked_as = get_womans_evaluation(graph, m, r);
                unsigned int woman_ranked_as = r;
                unsigned int womans_current_rank = womanCurrentRanking[f];

                if(man_ranked_as < womans_current_rank && woman_ranked_as < mans_current_rank)
                    correct = false;
            }


        }
        if(correct)
            printf("stable!\n");
        else
            printf("unstable!\n");

        bool *matched;
        if((error = posix_memalign((void **)&matched, 64, sizeof(bool)*graph->nrWoman)) != 0){
            printf("Could not allocate womanCurrentRanking: %d\n", error);
            exit(error);
        }
        for(size_t i = 0; i < graph->nrWoman; i++) {
            matched[i] = false;
        }
        printf("checking if noone is matched twice: ");
        bool matched_only_once = true;

        for(size_t i = 0; i < graph->nrMan && correct; i++) {
            unsigned int f = get_man_priority(graph, i, manCurrentRanking[i]);
            if(matched[f]) {
                printf("%d is already matched!\n", f);
                matched_only_once = false;
            } else {
                matched[f] = true;
            }
        }
        if(matched_only_once)
            printf("success!\n");
        else
            printf("failure!\n");

        correct &= matched_only_once;
        free(matched);

        printf("checking if everyone gets a match: \n");
        int match_count = calculate_match_count(graph, manCurrentRanking);

        if(graph->nrMan == match_count)
            printf("yes: %d pairs got a match\n", match_count);
        else
            printf("nope: %d pairs (%.3f%%) did not get a match\n", graph->nrMan - match_count, (1.0-(float)match_count / (float)graph->nrMan) * 100.0);

    free(womanCurrentRanking);
    free(manCurrentRanking);
    return correct;
}

void run_iterations(int iterations, CompleteBipartiteGraph *graph, omp_lock_t *w_locks, bool debug, double (*function)(CompleteBipartiteGraph *, omp_lock_t *, int *, int *, bool)) {

    int error;
    int *manCurrentRanking;
    int *womanCurrentRanking;

    if((error = posix_memalign((void **)&manCurrentRanking, 64, sizeof(int)*graph->nrMan)) != 0){
        printf("Could not allocate manCurrentRanking: %d\n", error);
        exit(error);
    }
    if((error = posix_memalign((void **)&womanCurrentRanking, 64, sizeof(int)*graph->nrWoman)) != 0){
        printf("Could not allocate womanCurrentRanking: %d\n", error);
        exit(error);
    }

    double *times;
    if((error = posix_memalign((void **)&times, 64, sizeof(double)*iterations)) != 0){
        printf("Could not allocate times: %d\n", error);
        exit(error);
    }
    printf("threads\tmean\tmedian\tmatches\tsum_ranking\n");
    fflush(stdout);

    int thread_counts_size = sizeof(thread_counts) / sizeof(thread_counts[0]);
    for(int t = 0; t < thread_counts_size; t++) {
        double sum = 0;
        int threads = thread_counts[t];
        omp_set_num_threads(threads);
        for(size_t i = 0; i < iterations; i++) {
            for(int64_t j = 0; j < graph->nrMan; j++) {
                manCurrentRanking[j] = -1;
            }
            for(int64_t j = 0; j < graph->nrWoman; j++) {
                womanCurrentRanking[j] = INT_MAX;
            }
            times[i] = (*function)(graph, w_locks, manCurrentRanking, womanCurrentRanking, debug);
            sum += times[i];
        }
        omp_set_num_threads(72);
        qsort(times, iterations, sizeof(double), compare_doubles);
        printf("%d\t%.3lf\t%.3lf", threads, sum/iterations, times[iterations/2]);
        fflush(stdout);

        unsigned int match_count = calculate_match_count(graph, manCurrentRanking);
        printf("\t%d", match_count);
        fflush(stdout);

        size_t edges_traversed = calculate_edges_traversed(graph, manCurrentRanking);
        printf("\t%ld\n", edges_traversed);
        fflush(stdout);
    }
    free(manCurrentRanking);
    free(womanCurrentRanking);

    free(times);
}

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


int main(int argc, char *argv[]) {

    bool debug = getenv("DEBUG");

    if(debug)
        printf("size of size_t is: %ld bits\n", sizeof(size_t)*8);
    if(debug)
        printf("size of bool is: %ld bits\n", sizeof(bool)*8);

    #pragma omp parallel
    {
        if(omp_get_thread_num() == 0 && debug)
            printf("threads: %d\n", omp_get_num_threads());
    }

    size_t iterations = 3;
    CompleteBipartiteGraph* complete_graph;
    int N;
    if(argc == 2){
        N = atoi(argv[1]);
        complete_graph = allocateComBipartite(N,N);
        makeCompleteBipartieGraph(complete_graph);
    } else if(argc == 3) {
        N = atoi(argv[1]);
        complete_graph = allocateComBipartite(N,N);
        readRanks(argv[2], complete_graph);
    } else {
        printf("No N number of men/women given\n");
        exit(1);
    }

    int error = 0;
    fflush(stdout);

    omp_lock_t *w_locks;
    if((error = posix_memalign((void **)&w_locks, 64, sizeof(omp_lock_t)*N)) != 0){
        printf("Could not allocate w_locks: %d\n", error);
        exit(error);
    }

    size_t l;
    for(l = 0; l < N; l++) {
        omp_init_lock(&w_locks[l]);
    }


    bool stack_correct = (!debug || run_correctness_check(complete_graph, w_locks, &experiment_stack));
    bool queue_correct = (!debug || run_correctness_check(complete_graph, w_locks, &experiment_queue));

    if(queue_correct && stack_correct) {
        if(debug)
            printf("Running iterations\n");

        printf("# stack: \n");
        run_iterations(iterations, complete_graph, w_locks, debug, &experiment_stack);
        printf("# queue: \n");
        run_iterations(iterations, complete_graph, w_locks, debug, &experiment_queue);

    }



    for(l = 0; l < N; l++) {
        omp_destroy_lock(&w_locks[l]);
    }


    free(w_locks);
    freeBipartite(complete_graph);
    fflush(stdout);



    return 0;
}
