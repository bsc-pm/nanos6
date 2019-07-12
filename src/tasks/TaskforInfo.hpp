/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKFOR_INFO_HPP
#define TASKFOR_INFO_HPP

#include <atomic>
#include <algorithm>

#include <nanos6.h>

#include "executors/threads/CPUManager.hpp"


struct TaskforInfo {
	typedef nanos6_loop_bounds_t bounds_t;
private:

	friend class Taskfor;

protected:
	bounds_t _bounds;
	size_t _maxCollaborators;
	std::atomic<size_t> _remainingIterations;
	size_t _completedIterations;

public:
	inline TaskforInfo(bool precreated)
		: _bounds(), _remainingIterations(0), _completedIterations(0)
	{
		_maxCollaborators = precreated ? 0 : CPUManager::getNumCPUsPerTaskforGroup();
		assert(precreated || CPUManager::getNumCPUsPerTaskforGroup() > 0);
		assert(precreated || _maxCollaborators > 0);
	}

	inline void reinitialize()
	{
		_bounds = bounds_t();
		_remainingIterations = 0;
		_completedIterations = 0;
	}

	inline ~TaskforInfo()
	{
	}

	inline void initialize()
	{
		assert(_maxCollaborators > 0);
		size_t totalIterations = getIterationCount();
		_remainingIterations = totalIterations;

		// Set a implementation defined chunksize if needed
		if (_bounds.chunksize == 0) {
			_bounds.chunksize = std::max(totalIterations / (_maxCollaborators), (size_t) 1);
		}
	}
	
	inline void initialize(size_t lowerBound, size_t upperBound, size_t chunksize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.chunksize = chunksize;

		initialize();
	}

	inline void initialize(const bounds_t &newBounds)
	{
		// Set the bounds
		_bounds.lower_bound = newBounds.lower_bound;
		_bounds.upper_bound = newBounds.upper_bound;
		_bounds.grainsize = newBounds.grainsize;
		_bounds.chunksize = newBounds.chunksize;
		
		initialize();
	}

	inline bounds_t &getBounds()
	{
		return _bounds;
	}

	inline bounds_t const &getBounds() const
	{
		return _bounds;
	}

	inline void setBounds(bounds_t &bounds)
	{
		_bounds.lower_bound = bounds.lower_bound;
		_bounds.upper_bound = bounds.upper_bound;
	}

	inline size_t getIterationCount()
	{
		return (_bounds.upper_bound - _bounds.lower_bound);
	}

	inline bool decrementRemainingIterations(size_t amount)
	{
		size_t remaining = (_remainingIterations -= amount);
		return (remaining == 0);
	}

	inline size_t getCompletedIterations()
	{
		return _completedIterations;
	}
};

#endif // TASKFOR_INFO_HPP
