/* 
 * File:   kernelsdef.h
 * Author: Naim
 *
 * Created on May 22, 2016, 6:51 PM
 */

#ifndef KERNELSDEF_H
#define	KERNELSDEF_H

#include "datadef.h"
#include "completeBipartite.h"



#ifdef RUNONGPU
__global__
#endif
void makeMatch(ManWomanGraph graph, unsigned int *currPtrsMen, unsigned int *suitorsRanks);

#ifdef RUNONGPU
__global__
#endif

void matchInCompleteBipartite(CompleteBipartiteGraph graph, unsigned int *currPtrsMen,
        unsigned int *suitorsRanks, unsigned int szNbrListForMen);
/*
#ifdef RUNONGPU
__global__
#endif
void printFromPtr(ManWomanGraph *graph);
 */
#endif	/* KERNELSDEF_H */

