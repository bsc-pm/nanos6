#ifndef COMPUTE_PLACE_CPP
#define COMPUTE_PLACE_CPP

#include "ComputePlace.hpp"
#include "MemoryPlace.hpp"

void ComputePlace::addMemoryPlace(MemoryPlace * mem) {
    _memoryPlaces[mem->getIndex()] = mem;
}

const std::vector<int>* ComputePlace::getMemoryPlacesIndexes(){
    std::vector<int>* indexes = new std::vector<int>();

    for(memoryPlaces_t::iterator it = _memoryPlaces.begin(); it != _memoryPlaces.end(); ++it){
        indexes->push_back(it->first);
    }

    return indexes;
}

const std::vector<MemoryPlace*>* ComputePlace::getMemoryPlaces(){
    std::vector<MemoryPlace*>* mems = new std::vector<MemoryPlace*>();

    for(memoryPlaces_t::iterator it = _memoryPlaces.begin(); it != _memoryPlaces.end(); ++it){
        mems->push_back(it->second);
    }

    return mems;
}

#endif //COMPUTE_PLACE_CPP
