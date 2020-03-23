/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESSES_INFO_HPP
#define TASK_DATA_ACCESSES_INFO_HPP

#include <cstdlib>

#include "DataAccess.hpp"

#define ACCESS_LINEAR_CUTOFF 256

class TaskDataAccessesInfo {
private:
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
		return _seqsSize + _addrSize;
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

		if (_seqsSize != 0)
			return reinterpret_cast<DataAccess *>(static_cast<char *>(_allocationAddress) + _addrSize);

		return nullptr;
	}

	inline size_t getNumDeps()
	{
		return _numDeps;
	}
};

#endif // TASK_DATA_ACCESSES_INFO_HPP