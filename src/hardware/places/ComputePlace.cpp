/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ComputePlace.hpp"
#include "MemoryPlace.hpp"

void ComputePlace::addMemoryPlace(MemoryPlace *mem) {
	_memoryPlaces[mem->getIndex()] = mem;
}

std::vector<int> ComputePlace::getMemoryPlacesIndexes() {
	std::vector<int> indexes(_memoryPlaces.size());
	
	int i = 0;
	for (memory_places_t::iterator it = _memoryPlaces.begin();
		it != _memoryPlaces.end();
		++it, ++i)
	{
		//indexes.push_back(it->first);
		indexes[i] = it->first;
	}
	
	return indexes;
}

std::vector<MemoryPlace *> ComputePlace::getMemoryPlaces() {
	std::vector<MemoryPlace *> mems(_memoryPlaces.size());
	
	int i = 0;
	for (memory_places_t::iterator it = _memoryPlaces.begin();
		it != _memoryPlaces.end();
		++it, ++i)
	{
		//mems.push_back(it->second);
		mems[i] = it->second;
	}
	
	return mems;
}
