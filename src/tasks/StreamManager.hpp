/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include <map>

#include <nanos6.h>

#include "StreamExecutor.hpp"
#include "system/ompss/SpawnFunction.hpp"


class StreamManager {

private:
	
	typedef std::map<size_t, StreamExecutor *> stream_executors_t;
	
	//! Singleton instance
	static StreamManager *_manager;
	
	//! Maps stream executors through their stream identifier
	stream_executors_t _executors;
	
	//! Spinlock to add new stream executors and access existent ones
	SpinLock _spinlock;
	
	
public:
	
	//! The amount of active stream executors
	static std::atomic<size_t> _activeStreamExecutors;
	
	
private:
	
	inline StreamManager() :
		_executors(),
		_spinlock()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	StreamManager(StreamManager const&) = delete;            // Copy construct
	StreamManager(StreamManager&&) = delete;                 // Move construct
	StreamManager& operator=(StreamManager const&) = delete; // Copy assign
	StreamManager& operator=(StreamManager &&) = delete;     // Move assign
	
	
	//    MANAGER    //
	
	//! \brief Initialize the manager
	static inline void initialize()
	{
		if (_manager == nullptr) {
			_manager = new StreamManager();
			assert(_manager != nullptr);
		}
	}
	
	//! \brief Shutdown mechanism for all the stream executors
	static inline void shutdown()
	{
		if (_manager != nullptr) {
			_manager->_spinlock.lock();
			
			// Notify all executors about the shutdown
			for (auto &it : _manager->_executors) {
				StreamExecutor *executor = it.second;
				assert(executor != nullptr);
				executor->notifyShutdown();
				delete it.second;
			}
			
			_manager->_spinlock.unlock();
			
			while (_activeStreamExecutors.load() > 0) {
				// Wait for all active stream executors to finalize
			}
		}
		
		delete _manager;
	}
	
	
	//    STREAM HANDLING    //
	
	//! \brief Create a function to be executed in a stream
	//! \param[in] function The function to execute
	//! \param[in] args Arguments of the function
	//! \param[in] label An optional label for the function
	//! \param[in] streamId The identifier of the stream
	static void createFunction(
		void (*function)(void *),
		void *args,
		char const *label,
		size_t streamId
	);
	
	//! \brief Synchronize (block control flow) of a certain stream until
	//! all spawned functions have finalized
	//! \param[in] streamId The identifier of the stream to synchronize
	static void synchronizeStream(size_t streamId);
	
};

#endif // STREAM_MANAGER_HPP
