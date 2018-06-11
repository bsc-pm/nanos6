/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_EVENT_POOL_HPP
#define CUDA_EVENT_POOL_HPP

#include "CUDAEvent.hpp"

#include <queue>

class CUDAEventPool {
	
private:
	std::queue<CUDAEvent *> _pool;
	
#ifndef NDEBUG
	int _poolSize;
#endif
	
public:
	
	CUDAEventPool()
	{
#ifndef NDEBUG
		_poolSize = 0;
#endif
	}
	
	~CUDAEventPool()
	{
		assert(_pool.size() == _poolSize);
		
		while (!_pool.empty()) {
			delete _pool.front();
			_pool.pop();
		}
	}
	
	CUDAEvent *getEvent()
	{
		if (_pool.empty()) {
#ifndef NDEBUG
			++_poolSize;
#endif
			return new CUDAEvent();	
		} else {
			CUDAEvent *event = _pool.front();
			_pool.pop();
			return event;
		}
	}
	
	void returnEvent(CUDAEvent *event)
	{
		_pool.push(event);
	}
};

#endif //CUDA_EVENT_POOL_HPP

