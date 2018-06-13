/* 
 * File:   graphIO.h
 * Author: frodo
 *
 * Created on May 23, 2016, 7:11 PM
 */

#ifndef GRAPHIO_H
#define	GRAPHIO_H

#include"datadef.h"
char* writeManWomanGraphInBinary(ManWomanGraph* graph);
ManWomanGraph* readManWomanGraphInBinary(const char* fileName);

#endif	/* GRAPHIO_H */

