/*
 * File:   datadef.h
 * Author: Naim
 *
 * Created on May 18, 2016, 1:50 PM
 */

#ifndef DATADEF_H
#define	DATADEF_H

#include <stdint.h>
typedef struct prob {
    unsigned int nrMan;
    unsigned int nrWoman;
    int64_t nrEdges;

    int64_t *indicesMan;
    int64_t *indicesWoman;

    unsigned int *mansPriorities;
    unsigned int *mansEvalbyWoman;

    unsigned int *womansPriorities;
} ManWomanGraph;

#endif	/* DATADEF_H */

