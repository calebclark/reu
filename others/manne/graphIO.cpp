#include"graphIO.h"
#include"memoryManager.h"
#include <ios>
#include <fstream>
#include <string.h>
#include"stdlib.h"
#include "iostream"
#include <sstream>

char* writeManWomanGraphInBinary(ManWomanGraph* graph) {

    std::ostringstream os ;

    os << "manWomanGraph_";
    os << graph->nrMan << "_";
    os << graph->nrWoman << "_";
    os << graph->nrEdges << ".dat";

    std::string fileName = os.str();

    std::cout << "Writing to " << fileName << std::endl;

    std::ofstream ofs;
    ofs.open(fileName.c_str(), std::ios::out | std::ios::binary);

    if (graph) {
        ofs.write((char*) &graph->nrMan, sizeof (unsigned int));
        ofs.write((char*) &graph->nrWoman, sizeof (unsigned int));
        ofs.write((char*) &graph->nrEdges, sizeof (int64_t));
        if (graph->indicesMan && graph->indicesWoman && graph->mansPriorities
                && graph->mansEvalbyWoman && graph->womansPriorities) {

            ofs.write((char*) &graph->indicesMan[0], sizeof (int64_t)*(graph->nrMan + 1));
            ofs.write((char*) &graph->indicesWoman[0], sizeof (int64_t)*(graph->nrWoman + 1));

            ofs.write((char*) &graph->mansPriorities[0], sizeof (unsigned int) * graph->nrEdges);
            ofs.write((char*) &graph->mansEvalbyWoman[0], sizeof (unsigned int) * graph->nrEdges);
            ofs.write((char*) &graph->womansPriorities[0], sizeof (unsigned int) * graph->nrEdges);
        }
    }
    ofs.close();

    char* nameToReturn = (char*) malloc(sizeof (char)*(fileName.size() + 1));
    strcpy(nameToReturn, fileName.c_str());

    printf("Wrote to %s\n", fileName.c_str());
    return nameToReturn;
}

ManWomanGraph* readManWomanGraphInBinary(const char* fileName) {


    ManWomanGraph* graph = NULL;

    std::ifstream ifs;
    ifs.open(fileName, std::ios::in | std::ios::binary);
    if (ifs.is_open()) {

        unsigned int nrMan;
        unsigned int nrWoman;
        int64_t nrEdges;

        ifs.read((char*) &nrMan, sizeof (unsigned int));
        ifs.read((char*) &nrWoman, sizeof (unsigned int));
        ifs.read((char*) &nrEdges, sizeof (int64_t));

        //Allocate memory for the graph
        graph = allocateGraph(nrMan, nrWoman, nrEdges);


        printf("\nReading File with %d Men, %d Women and %lld Edges\n", graph->nrMan, graph->nrWoman, graph->nrEdges);

        if (graph->indicesMan && graph->indicesWoman && graph->mansPriorities
                && graph->mansEvalbyWoman && graph->womansPriorities) {

            ifs.read((char*) &graph->indicesMan[0], sizeof (int64_t)*(graph->nrMan + 1));
            ifs.read((char*) &graph->indicesWoman[0], sizeof (int64_t)*(graph->nrWoman + 1));

            ifs.read((char*) &graph->mansPriorities[0], sizeof (unsigned int) * graph->nrEdges);
            ifs.read((char*) &graph->mansEvalbyWoman[0], sizeof (unsigned int) * graph->nrEdges);

            ifs.read((char*) &graph->womansPriorities[0], sizeof (unsigned int) * graph->nrEdges);
        } else {
            printf("ERROR: One of the array in NULL\n");
        }
    } else {
        printf("\nERROR:Can't open inputGraph\n");
    }
    return graph;
}
