/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NUMA_PLACE_CPP
#define NUMA_PLACE_CPP

#include "NUMAPlace.hpp"
#include "ComputePlace.hpp"

void NUMAPlace::addComputePlace(ComputePlace * computePlace) {
	_computePlaces[computePlace->getIndex()] = computePlace;
}

std::vector<int> NUMAPlace::getComputePlaceIndexes(){
	//! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<int> indexes(_computePlaces.size());

	//! Double iterator needed to overwrite the already initialized positions of the vector.
	int i = 0;
	for(computePlaces_t::iterator it = _computePlaces.begin(); 
		it != _computePlaces.end(); 
		++it, ++i)
	{
		indexes[i] = it->first;
	}

	return indexes;
}

std::vector<ComputePlace*> NUMAPlace::getComputePlaces(){
	//! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<ComputePlace*> computePlaces(_computePlaces.size());

	//! Double iterator needed to overwrite the already initialized positions of the vector.
	int i = 0;
	for(computePlaces_t::iterator it = _computePlaces.begin(); 
		it != _computePlaces.end(); 
		++it, ++i)
	{
		computePlaces[i] = it->second;
	}

	return computePlaces;
}

#endif //NUMA_PLACE_CPP
