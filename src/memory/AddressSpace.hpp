/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <cstddef>
#include <vector>
#include <map>

class MemoryPlace;

class AddressSpace {
private:
	typedef std::map<int, MemoryPlace*> memory_places_t;
	memory_places_t _memoryPlaces; // MemoryPlaces within this AddressSpace
	
public:
	AddressSpace()
	{}
	
	virtual ~AddressSpace()
	{}
	
	size_t getMemoryPlacesCount() const
	{
		return _memoryPlaces.size();
	}
	
	MemoryPlace *getMemoryPlace(unsigned int index)
	{
		return _memoryPlaces[index];
	}
	
	void addMemoryPlace(MemoryPlace *memoryPlace);
	
	std::vector<unsigned int> getMemoryPlacesIndexes() const;
	
	std::vector<MemoryPlace*> getMemoryPlaces() const;
};

#endif //ADDRESS_SPACE_HPP
