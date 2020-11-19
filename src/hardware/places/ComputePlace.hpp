/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP

#include <map>
#include <random>
#include <vector>

#include <nanos6/task-instantiation.h>

#include "CPUDependencyData.hpp"

#include <InstrumentComputePlaceId.hpp>
#include <InstrumentCPULocalData.hpp>

class Taskfor;
class MemoryPlace;

//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class ComputePlace {
private:
	typedef std::map<int, MemoryPlace *> memory_places_t;

	//! Accessible from this compute place
	memory_places_t _memoryPlaces;

	//! Preallocated taskfor to be used as taskfor collaborator
	Taskfor *_preallocatedTaskfor;

	//! Preallocated argsBlock for the taskfor collaborator
	void *_preallocatedArgsBlock;

	//! The size of the preallocated argsBlock
	size_t _preallocatedArgsBlockSize;

	//! Whether this cpu is owned by the runtime
	bool _owned;

	//! Random generator. Currently used in TaskDataAccesses::computeNUMAAffinity
	std::default_random_engine _randomEngine;

protected:
	//! The index of the compute place
	size_t _index;

	//! The device type of the compute place
	nanos6_device_t _type;

	//! The instrumentation id of the compute place
	Instrument::compute_place_id_t _instrumentationId;

	//! Per CPU Instrumentation data
	Instrument::CPULocalData _instrumentationData;

	//! The dependency data for this compute place
	CPUDependencyData _dependencyData;

public:
	ComputePlace(int index, nanos6_device_t type, bool owned = true);

	virtual ~ComputePlace();

	inline size_t getMemoryPlacesCount() const
	{
		return _memoryPlaces.size();
	}

	inline MemoryPlace *getMemoryPlace(int index)
	{
		memory_places_t::iterator it = _memoryPlaces.find(index);
		if (it != _memoryPlaces.end()) {
			return it->second;
		}
		return nullptr;
	}

	//! \brief returns the preallocated taskfor of this ComputePlace.
	inline Taskfor *getPreallocatedTaskfor()
	{
		return _preallocatedTaskfor;
	}

	void *getPreallocatedArgsBlock(size_t requiredSize);

	inline int getIndex() const
	{
		return _index;
	}

	inline void setIndex(int index)
	{
		_index = index;
	}

	inline nanos6_device_t getType() const
	{
		return _type;
	}

	inline bool isOwned() const
	{
		return _owned;
	}

	inline void setOwned(bool owned = true)
	{
		_owned = owned;
	}

	void addMemoryPlace(MemoryPlace *mem);

	std::vector<int> getMemoryPlacesIndexes();

	std::vector<MemoryPlace *> getMemoryPlaces();

	inline void setInstrumentationId(Instrument::compute_place_id_t const &instrumentationId)
	{
		_instrumentationId = instrumentationId;
	}

	inline Instrument::compute_place_id_t const &getInstrumentationId() const
	{
		return _instrumentationId;
	}

	inline CPUDependencyData &getDependencyData()
	{
		return _dependencyData;
	}

	Instrument::CPULocalData &getInstrumentationData()
	{
		return _instrumentationData;
	}

	inline std::default_random_engine &getRandomEngine()
	{
		return _randomEngine;
	}
};

#endif //COMPUTE_PLACE_HPP
