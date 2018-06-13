/* 
 * File:   graphChecker.h
 * Author: frodo
 *
 * Created on May 23, 2016, 7:17 PM
 */

#ifndef GRAPHCHECKER_H
#define	GRAPHCHECKER_H

#include"datadef.h"


bool checkManWomanGraph(ManWomanGraph* graph, int nrThreads);
void displayGraph(ManWomanGraph* graph);
void displaySparse(ManWomanGraph* graph);

#endif	/* GRAPHCHECKER_H */

