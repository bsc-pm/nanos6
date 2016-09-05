#ifndef MEMORY_PLACE_CPP
#define MEMORY_PLACE_CPP

#include "MemoryPlace.hpp"
#include "ComputePlace.hpp"

void MemoryPlace::addPU(ComputePlace * pu) {
    _processingUnits[pu->getIndex()] = pu;
}

const std::vector<int>* MemoryPlace::getPUIndexes(){
    std::vector<int>* indexes = new std::vector<int>();

    for(processingUnits_t::iterator it = _processingUnits.begin(); it != _processingUnits.end(); ++it){
        indexes->push_back(it->first);
    }

    return indexes;
}

const std::vector<ComputePlace*>* MemoryPlace::getPUs(){
    std::vector<ComputePlace*>* PUs = new std::vector<ComputePlace*>();

    for(processingUnits_t::iterator it = _processingUnits.begin(); it != _processingUnits.end(); ++it){
        PUs->push_back(it->second);
    }

    return PUs;
}

#endif //MEMORY_PLACE_CPP
