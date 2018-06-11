/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_STREAM_POOL_HPP
#define CUDA_STREAM_POOL_HPP

#define CUDA_STARTING_STREAM_NUM 0

#include <queue>

#include "CUDAStream.hpp"

class CUDAStreamPool {
private:
	std::queue<CUDAStream *> _pool;
#ifndef NDEBUG
	int _poolSize; /*! Debbuging counter to check that all streams are cleared from the system on shutdown */
#endif
	
public:
	
	CUDAStreamPool()
	{
		for (int i = 0; i < CUDA_STARTING_STREAM_NUM; ++i) {
			_pool.push(new CUDAStream());
		}
#ifndef NDEBUG
		_poolSize = CUDA_STARTING_STREAM_NUM;
#endif
	}
	
	~CUDAStreamPool()
	{
		assert(_pool.size() == _poolSize);
		
		while (!_pool.empty()) {
			delete _pool.front();
			_pool.pop();
		}
	}
	
	/*
	 * \!brief Get a CUDA stream
	 *
	 * Get a CUDA stream from the pool. 
	 * If no streams are available a new stream is returned, which will be eventually returned to the pool instead of released.
	 */
	CUDAStream *getStream() 
	{
		if (_pool.empty()) {
#ifndef NDEBUG
			++_poolSize;
#endif
			return new CUDAStream();
		} else {
			CUDAStream *stream = _pool.front();
			_pool.pop();
			return stream;
		}
	}
	/*
	 *	\!brief Return a CUDA stream to the pool
	 */
	void returnStream(CUDAStream *stream)
	{
		_pool.push(stream);	
	}
};


#endif //CUDA_STREAM_POOL_HPP

