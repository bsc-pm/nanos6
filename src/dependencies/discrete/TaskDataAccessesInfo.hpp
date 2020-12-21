/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESSES_INFO_HPP
#define TASK_DATA_ACCESSES_INFO_HPP

#include <cstdlib>

#include "DataAccess.hpp"
#include "lowlevel/Padding.hpp"

#define ACCESS_LINEAR_CUTOFF 256

class TaskDataAccessesInfo {
private:
	static constexpr size_t _alignSize = CACHELINE_SIZE - 1;
	size_t _numDeps;
	size_t _seqsSize;
	size_t _addrSize;

	void *_allocationAddress;

public:
	TaskDataAccessesInfo(size_t numDeps) :
		_numDeps(numDeps), _seqsSize(0), _addrSize(0), _allocationAddress(nullptr)
	{
		if (numDeps <= ACCESS_LINEAR_CUTOFF) {
			_seqsSize = sizeof(DataAccess) * numDeps;
			_addrSize = sizeof(void *) * numDeps;
		}
	}

	inline size_t getAllocationSize()
	{
		return _seqsSize + _addrSize + (_numDeps > 0 ? _alignSize : 0);
	}

	inline void setAllocationAddress(void *allocationAddress)
	{
		_allocationAddress = allocationAddress;
	}

	inline void **getAddressArrayLocation()
	{
		assert(_allocationAddress != nullptr || _numDeps == 0);

		if (_addrSize != 0)
			return static_cast<void **>(_allocationAddress);

		return nullptr;
	}

	inline DataAccess *getAccessArrayLocation()
	{
		assert(_allocationAddress != nullptr || _numDeps == 0);

		if (_seqsSize != 0) {
			// We need an integral type to perform the modulo operations
			uintptr_t addrLocation = reinterpret_cast<uintptr_t>(_allocationAddress) + _addrSize;

			// We must align the DataAccesses to a cacheline to prevent false sharing.
			if (addrLocation % CACHELINE_SIZE) {
				// Add padding
				addrLocation += CACHELINE_SIZE - (addrLocation % CACHELINE_SIZE);
				assert(addrLocation % CACHELINE_SIZE == 0);
			}

			return reinterpret_cast<DataAccess *>(addrLocation);
		}

		return nullptr;
	}

	inline size_t getNumDeps()
	{
		return _numDeps;
	}
};

#endif // TASK_DATA_ACCESSES_INFO_HPP
