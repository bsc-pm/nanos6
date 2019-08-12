/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_CONTEXT_HPP
#define CUDA_CONTEXT_HPP

#include <cstddef>
#include <queue>

#include <cuda_runtime_api.h>

#include "lowlevel/cuda/CUDAErrorHandler.hpp"
#include "tasks/Task.hpp"

#define CUDA_STARTING_STREAM_NUM 0

class CUDAStream {
private:
	size_t _index;
	cudaStream_t _stream;
	
public:
	CUDAStream(size_t index) :
		_index(index)
	{
		cudaError_t err = cudaStreamCreate(&_stream);
		CUDAErrorHandler::handle(err, "When creating stream");
	}
	
	// Disable copy constructor
	CUDAStream(CUDAStream const&) = delete;
	CUDAStream operator=(CUDAStream const&) = delete;
	
	~CUDAStream()
	{
		cudaError_t err = cudaStreamDestroy(_stream);
		CUDAErrorHandler::handle(err, "When destroying stream");
	}
	
	//! \brief Get the assigned index of the stream
	size_t getIndex() const
	{
		return _index;
	}
	
	//! \brief Get the underlying cudaStream_t object
	cudaStream_t getStream() const
	{
		return _stream;
	}
};

class CUDAEvent {
private:
	Task *_task;
	cudaEvent_t _event;
	
public:
	CUDAEvent() : _task(nullptr)
	{
		cudaError_t err = cudaEventCreateWithFlags(&_event, cudaEventDisableTiming);
		CUDAErrorHandler::handle(err, "When creating event");
	}
	
	~CUDAEvent()
	{
		cudaError_t err = cudaEventDestroy(_event);
		CUDAErrorHandler::handle(err, "When destroying event");
	}
	
	void setTask(Task *task)
	{
		_task = task;
	}
	Task *getTask()
	{
		return _task;
	}
	
	void record()
	{
		cudaError_t err = cudaEventRecord(_event,
				((CUDAStream *) (_task->getDeviceData()))->getStream());
		
		CUDAErrorHandler::handle(err, " When recording event, fml");
	}
	
	bool finished()
	{
		cudaError_t err = cudaEventQuery(_event);
		return CUDAErrorHandler::handleEvent(err, " When checking event status");
	}
};

class CUDAStreamPool {
private:
	std::queue<CUDAStream *> _pool;
	size_t _size;
	
public:
	CUDAStreamPool()
	{
		for (int i = 0; i < CUDA_STARTING_STREAM_NUM; ++i) {
			_pool.push(new CUDAStream(i));
		}
		_size = CUDA_STARTING_STREAM_NUM;
	}
	
	~CUDAStreamPool()
	{
		assert(_pool.size() == _size);
		
		while (!_pool.empty()) {
			delete _pool.front();
			_pool.pop();
		}
	}
	
	//!	\!brief Get a CUDA stream
	//!	Get a CUDA stream from the pool.
	//!	If no streams are available a new stream is returned, which will be
	//! eventually returned to the pool instead of released.
	CUDAStream *getStream()
	{
		if (_pool.empty()) {
			++_size;
			return new CUDAStream(_size - 1);
		}
		else {
			CUDAStream *stream = _pool.front();
			assert(stream != nullptr);
			_pool.pop();
			return stream;
		}
	}
	//!	\!brief Return a CUDA stream to the pool
	void returnStream(CUDAStream *stream)
	{
		_pool.push(stream);
	}
};

class CUDAEventPool {
private:
	std::queue<CUDAEvent *> _pool;
	
#ifndef NDEBUG
	size_t _poolSize;
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
		}
		else {
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

struct CUDA_DEVICE_DEP {
	CUDAStreamPool _pool;
	CUDAEventPool _eventPool;
	std::vector<CUDAEvent *> _activeEvents;
};

#endif //CUDA_CONTEXT_HPP
