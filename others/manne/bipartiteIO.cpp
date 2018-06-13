
#include <ios>
#include <fstream>
#include <string.h>
#include"string"
#include"stdlib.h"
#include "iostream"
#include"unordered_set"
#include"completeBipartite.h"
#include"bipartiteIO.h"
#include <sstream>
using namespace std;

/**
 * @param nrToGenerate
 * @param rankPtr Make sure rankPtr is allocated beforehand
 */
static void generateRanks(unsigned int nrToGenerate, unsigned int *rankPtr) {


    srand(int(time(NULL)));

    std::unordered_set<int> myset;
    std::unordered_set<int>::iterator it;

    while (myset.size() < nrToGenerate) {
        int rint = rand() % nrToGenerate;
        myset.insert(rint);
    }

    if (nrToGenerate != myset.size())
        std::cout << "ERROR: Don't have enough choices for Man " << std::endl;

    it = myset.begin();

    for (int j = 0; j < nrToGenerate; j++) {
        rankPtr[j] = *it;
        it++;
    }

    myset.clear();
}

char* writeRanks(CompleteBipartiteGraph* graph) {


    std::ostringstream os ;

    os << "ranking_";
    os << graph->nrMan << "_";
    os << graph->nrWoman << ".dat";

    std::string fileName = os.str();

    std::cout << "Writing to " << fileName << std::endl;

    std::ofstream ofs;
    ofs.open(fileName.c_str(), std::ios::out | std::ios::binary);

    if (graph) {
        ofs.write((char*) &graph->nrMan, sizeof (unsigned int));
        ofs.write((char*) &graph->nrWoman, sizeof (unsigned int));

        if (graph->commonRanksOfMen && graph->womenPeferredByMen) {

            // generate rank of each man
            generateRanks(graph->nrMan, graph->commonRanksOfMen);
            // write these
            ofs.write((char*) &graph->commonRanksOfMen[0], sizeof (unsigned int) * graph->nrMan);

            // generate rank of each woman
            generateRanks(graph->nrWoman, graph->womenPeferredByMen);
            // write these
            ofs.write((char*) &graph->womenPeferredByMen[0], sizeof (unsigned int) * graph->nrWoman);
        }
    }
    ofs.close();

    char* nameToReturn = (char*) malloc(sizeof (char) * (fileName.size() + 1));

    strcpy(nameToReturn, fileName.c_str());
    printf("Wrote to %s\n", fileName.c_str());

    return nameToReturn;
}

/**
 * Make sure graph is already allocated
 * @param fileName
 * @param graph
 */
void readRanks(const char* fileName, CompleteBipartiteGraph* graph) {

    std::ifstream ifs;
    ifs.open(fileName, std::ios::in | std::ios::binary);
    if (ifs.is_open()) {

        unsigned int nrMan;
        unsigned int nrWoman;
        unsigned int nrEdges;

        ifs.read((char*) &nrMan, sizeof (unsigned int));
        ifs.read((char*) &nrWoman, sizeof (unsigned int));

        printf("\nReading File with %d Men, %d Women \n", graph->nrMan, graph->nrWoman);

        if (graph->commonRanksOfMen && graph->womenPeferredByMen) {

            ifs.read((char*) &graph->commonRanksOfMen[0], sizeof (unsigned int) * graph->nrMan);

            for (int i = 0; i < graph->nrMan; i++) {
                graph->rankToManId[ graph->commonRanksOfMen[i]] = i;
            }

            ifs.read((char*) &graph->womenPeferredByMen[0], sizeof (unsigned int) * graph->nrWoman);

        } else {
            printf("ERROR: One of the array in NULL\n");
        }
    } else {
        printf("\nERROR:Can't open inputGraph\n");
    }
}

void displayBipartite(CompleteBipartiteGraph* graph) {
    if (graph) {

        if (graph->commonRanksOfMen)
            for (int i = 0; i < graph->nrMan; i++) {
                printf("%u ", graph->commonRanksOfMen[i]);
            }

        printf("\n");

        if (graph->womenPeferredByMen)
            for (int i = 0; i < graph->nrWoman; i++) {
                printf("%u ", graph->womenPeferredByMen[i]);
            }
        printf("\n");
    } else {
        printf("NULL Graph\n");
    }
}
