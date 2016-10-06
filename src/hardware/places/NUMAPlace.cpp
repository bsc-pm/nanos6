#ifndef NUMA_PLACE_CPP
#define NUMA_PLACE_CPP

#include "NUMAPlace.hpp"
#include "ComputePlace.hpp"

void NUMAPlace::addPU(ComputePlace * pu) {
    _processingUnits[pu->getIndex()] = pu;
}

const std::vector<int>* NUMAPlace::getPUIndexes(){
    std::vector<int>* indexes = new std::vector<int>();

    for(processingUnits_t::iterator it = _processingUnits.begin(); it != _processingUnits.end(); ++it){
        indexes->push_back(it->first);
    }

    return indexes;
}

const std::vector<ComputePlace*>* NUMAPlace::getPUs(){
    std::vector<ComputePlace*>* PUs = new std::vector<ComputePlace*>();

    for(processingUnits_t::iterator it = _processingUnits.begin(); it != _processingUnits.end(); ++it){
        PUs->push_back(it->second);
    }

    return PUs;
}

#endif //NUMA_PLACE_CPP
