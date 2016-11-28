#ifndef NUMA_PLACE_CPP
#define NUMA_PLACE_CPP

#include "NUMAPlace.hpp"
#include "ComputePlace.hpp"

void NUMAPlace::addPU(ComputePlace * pu) {
    _processingUnits[pu->getIndex()] = pu;
}

std::vector<int> NUMAPlace::getPUIndexes(){
    std::vector<int> indexes(_processingUnits.size());

    int i = 0;
    for(processingUnits_t::iterator it = _processingUnits.begin(); 
        it != _processingUnits.end(); 
        ++it, ++i)
    {
        indexes[i] = it->first;
        //indexes.push_back(it->first);
    }

    return indexes;
}

std::vector<ComputePlace*> NUMAPlace::getPUs(){
    std::vector<ComputePlace*> PUs(_processingUnits.size());

    int i = 0;
    for(processingUnits_t::iterator it = _processingUnits.begin(); 
        it != _processingUnits.end(); 
        ++it, ++i)
    {
        PUs[i] = it->second;
        //PUs.push_back(it->second);
    }

    return PUs;
}

#endif //NUMA_PLACE_CPP
