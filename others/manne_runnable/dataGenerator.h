/* 
 * File:   dataGenerator.h
 * Author: frodo
 *
 * Created on May 23, 2016, 7:18 PM
 */

#ifndef DATAGENERATOR_H
#define	DATAGENERATOR_H

#include"datadef.h"
ManWomanGraph* generateGraph(unsigned int nrMan, unsigned int nrWoman, unsigned int nrThreads);
ManWomanGraph* generateGraph_Seq(unsigned int nrMan, unsigned int nrWoman);
#endif	/* DATAGENERATOR_H */

