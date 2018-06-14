/*
 * File:   memoryManager.h
 * Author: frodo
 *
 * Created on May 23, 2016, 7:19 PM
 */

#ifndef MEMORYMANAGER_H
#define	MEMORYMANAGER_H
#include"datadef.h"
#include"completeBipartite.h"

ManWomanGraph* allocateGraph(unsigned int nrMan, unsigned int nrWoman, int64_t nrEdges);
void freeGraph(ManWomanGraph* graph);

CompleteBipartiteGraph* allocateComBipartite(unsigned int nrMan, unsigned int nrWoman);
void freeBipartite(CompleteBipartiteGraph* graph);



#endif	/* MEMORYMANAGER_H */

