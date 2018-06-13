/*
 * File:   bipartiteIO.h
 * Author: frodo
 *
 * Created on August 11, 2016, 12:45 PM
 */

#ifndef BIPARTITEIO_H
#define	BIPARTITEIO_H

#include "completeBipartite.h"

char* writeRanks(CompleteBipartiteGraph* graph);
void readRanks(const char* fileName, CompleteBipartiteGraph* graph);
void displayBipartite(CompleteBipartiteGraph* graph);


#endif	/* BIPARTITEIO_H */

