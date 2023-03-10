/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <cassert>

#include <limits.h>

#include <nanos6/task-instantiation.h>

#include "DataAccessFlags.hpp"
#include "support/Containers.hpp"

class Task;

class TaskList {
public:
	static size_t _actualChunkSize;

private:
	static constexpr size_t MAX_CHUNKSIZE = 256;

	Task *_array[MAX_CHUNKSIZE];

	size_t _count;

public:
	inline TaskList() :
		_count(0)
	{
	}

	inline size_t size() const
	{
		return _count;
	}

	inline void clear()
	{
		_count = 0;
	}

	inline Task *get(size_t pos)
	{
		assert(pos < _count);
		return _array[pos];
	}

	inline void add(Task *task)
	{
		assert(_count < _actualChunkSize);
		_array[_count++] = task;
	}

	inline void erase(size_t pos)
	{
		assert(pos < _count);

		// Move the last element (if any) to the erased position
		if (pos < _count - 1)
			_array[pos] = _array[_count - 1];

		--_count;
	}

	inline Task **getArray()
	{
		return &_array[0];
	}

	static inline size_t getMaxChunkSize()
	{
		return MAX_CHUNKSIZE;
	}
};

struct CPUDependencyData {
	typedef TaskList task_list_t;
	typedef Container::deque<Task *> commutative_satisfied_list_t;

	//! Tasks whose accesses have been satisfied after ending a task
	task_list_t _satisfiedOriginators[nanos6_device_t::nanos6_device_type_num];
	task_list_t _deletableOriginators;

	size_t _satisfiedOriginatorCount;

	commutative_satisfied_list_t _satisfiedCommutativeOriginators;
	mailbox_t _mailBox;

	size_t *_bytesInNUMA;

#ifndef NDEBUG
	std::atomic<bool> _inUse;
#endif

	CPUDependencyData()
		: _satisfiedOriginators(),
		_deletableOriginators(),
		_satisfiedOriginatorCount(0),
		_satisfiedCommutativeOriginators(),
		_mailBox(),
		_bytesInNUMA(nullptr)
#ifndef NDEBUG
		, _inUse()
#endif
	{
	}

	~CPUDependencyData()
	{
		assert(empty());

		if (_bytesInNUMA != nullptr) {
			std::free(_bytesInNUMA);
		}
	}

	inline bool empty() const
	{
		for (const task_list_t &list : _satisfiedOriginators)
			if (list.size() > 0)
				return false;

		if (_deletableOriginators.size() > 0)
			return false;

		return _mailBox.empty() && _satisfiedCommutativeOriginators.empty();
	}

	inline void addSatisfiedOriginator(Task *task, int deviceType)
	{
		assert(task != nullptr);
		assert(_satisfiedOriginatorCount < (size_t) task_list_t::_actualChunkSize);
		_satisfiedOriginatorCount++;
		_satisfiedOriginators[deviceType].add(task);
	}

	inline void addDeletableOriginator(Task *task)
	{
		assert(task != nullptr);
		assert(_deletableOriginators.size() < (size_t) task_list_t::_actualChunkSize);
		_deletableOriginators.add(task);
	}

	inline bool fullSatisfiedOriginators() const
	{
		assert(task_list_t::_actualChunkSize != 0);
		return (_satisfiedOriginatorCount == task_list_t::_actualChunkSize);
	}

	inline bool fullDeletableOriginators() const
	{
		assert(task_list_t::_actualChunkSize != 0);
		return (_deletableOriginators.size() == task_list_t::_actualChunkSize);
	}

	inline task_list_t &getSatisfiedOriginators(int device)
	{
		return _satisfiedOriginators[device];
	}

	inline task_list_t &getDeletableOriginators()
	{
		return _deletableOriginators;
	}

	inline void clearSatisfiedOriginators()
	{
		for (task_list_t &list : _satisfiedOriginators)
			list.clear();

		_satisfiedOriginatorCount = 0;
	}

	inline void clearDeletableOriginators()
	{
		_deletableOriginators.clear();
	}

	inline void initBytesInNUMA(int numNUMANodes)
	{
		if (_bytesInNUMA == nullptr) {
			_bytesInNUMA = (size_t *) std::malloc(sizeof(size_t) * numNUMANodes);
			assert(_bytesInNUMA != nullptr);
		}
	}
};


#endif // CPU_DEPENDENCY_DATA_HPP
