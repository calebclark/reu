/* 
 * File:   hostDeviceMemoryManager.h
 * Author: frodo
 *
 * Created on May 24, 2016, 4:21 PM
 */

#ifndef HOSTDEVICEMEMORYMANAGER_H
#define	HOSTDEVICEMEMORYMANAGER_H

#include "datadef.h"
#include"completeBipartite.h"

ManWomanGraph* allocateGraphOnDev(ManWomanGraph* hostGraph);

void copyGraphH2D(ManWomanGraph* hostGraph, ManWomanGraph* devGraph);

void freeManWomanDevGraph(ManWomanGraph* devGraph);

void allocateDevMem(void **memPtr, size_t nrBytes);
void copyArray(void* dest, void *source, size_t nrBytes, cudaMemcpyKind direction);

void freeDevMem(void *memPtr);


CompleteBipartiteGraph* allocateAndCopyBipartieGraphOnDev(CompleteBipartiteGraph* graph);
void freeCompleteBipartiteDevGraph(CompleteBipartiteGraph* devGraph);

#endif	/* HOSTDEVICEMEMORYMANAGER_H */

