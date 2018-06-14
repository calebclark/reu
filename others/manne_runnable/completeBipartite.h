/* 
 * File:   completeBipartite.h
 * Author: frodo
 *
 * Created on May 30, 2016, 5:52 PM
 */

#ifndef COMPLETEBIPARTITE_H
#define	COMPLETEBIPARTITE_H

typedef struct {
    unsigned int nrMan;
    unsigned int nrWoman;
    unsigned int* commonRanksOfMen;
    unsigned int* rankToManId;
    unsigned int* womenPeferredByMen;

} CompleteBipartiteGraph;



#endif	/* COMPLETEBIPARTITE_H */

