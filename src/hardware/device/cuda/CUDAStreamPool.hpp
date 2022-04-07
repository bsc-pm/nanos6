/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_STREAM_POOL_HPP
#define CUDA_STREAM_POOL_HPP

#include <queue>

#include "CUDAFunctions.hpp"

// For each CUDA device task a CUDA stream is required for the asynchronous
// launch; To ensure kernel completion a CUDA event is 'recorded' on the stream
// right after the kernel is queued. Then when a cudaEventQuery call returns
// succefully, we can be sure that the kernel execution (and hence the task)
// has finished.
// The class has been extended to also contain the associated CUDA events.
class CUDAStreamPool {
private:
	std::queue<cudaStream_t> _streams;
	std::queue<cudaEvent_t> _events;

public:
	CUDAStreamPool(int device)
	{
		static ConfigVariable<int> maxStreams("devices.cuda.streams");
		CUDAFunctions::setActiveDevice(device);
		for (int i = 0; i < maxStreams; ++i)
			_streams.emplace(CUDAFunctions::createStream());
	}

	~CUDAStreamPool()
	{
		while (!_events.empty()) {
			CUDAFunctions::destroyEvent(_events.front());
			_events.pop();
		}

		while (!_streams.empty()) {
			CUDAFunctions::destroyStream(_streams.front());
			_streams.pop();
		}
	}

	bool streamAvailable()
	{
		return !_streams.empty();
	}

	cudaStream_t getCUDAStream()
	{
		assert(!_streams.empty());
		cudaStream_t stream = _streams.front();
		_streams.pop();
		return stream;
	}

	// Release used stream of finished task back to the pool for future use
	void releaseCUDAStream(cudaStream_t stream)
	{
		_streams.emplace(stream);
	}

	cudaEvent_t getCUDAEvent()
	{
		cudaEvent_t event;
		if (_events.empty()) {
			CUDAFunctions::createEvent(event);
		} else {
			event = _events.front();
			_events.pop();
		}
		return event;
	}

	// Release used event of finished task back to the pool for future use
	void releaseCUDAEvent(cudaEvent_t event)
	{
		_events.emplace(event);
	}

};

#endif // CUDA_STREAM_POOL_HPP
