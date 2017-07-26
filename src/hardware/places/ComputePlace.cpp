/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMPUTE_PLACE_CPP
#define COMPUTE_PLACE_CPP

#include "ComputePlace.hpp"
#include "MemoryPlace.hpp"

void ComputePlace::addMemoryPlace(MemoryPlace * mem) {
	_memoryPlaces[mem->getIndex()] = mem;
}

std::vector<int> ComputePlace::getMemoryPlacesIndexes() {
	std::vector<int> indexes(_memoryPlaces.size());

	int i = 0;
	for(memoryPlaces_t::iterator it = _memoryPlaces.begin(); 
		it != _memoryPlaces.end(); 
		++it, ++i)
	{
		//indexes.push_back(it->first);
		indexes[i] = it->first;
	}

	return indexes;
}

std::vector<MemoryPlace*> ComputePlace::getMemoryPlaces() {
	std::vector<MemoryPlace*> mems(_memoryPlaces.size());

	int i = 0;
	for(memoryPlaces_t::iterator it = _memoryPlaces.begin(); 
		it != _memoryPlaces.end(); 
		++it, ++i)
	{
		//mems.push_back(it->second);
		mems[i] = it->second;
	}

	return mems;
}

#endif //COMPUTE_PLACE_CPP
