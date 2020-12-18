/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_LOCALITY_HPP
#define READY_QUEUE_LOCALITY_HPP

#include <vector>

#include "hardware/HardwareInfo.hpp"
#include "scheduling/SchedulerSupport.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "tasks/Task.hpp"

class ReadyQueueLocality : public ReadyQueue {
	typedef SchedulerSupport::LocalityQueue ready_queue_t;
	typedef std::deque<Task *> main_ready_queue_t;

	ready_queue_t *_L2Queues;
	ready_queue_t *_L3Queues;
	main_ready_queue_t *_NUMAQueues;
	main_ready_queue_t _mainQueue;

	size_t _numL2Queues;
	size_t _numL3Queues;
	uint8_t _numNUMAQueues;

	size_t _numReadyTasks;
	size_t _numL2Tasks;
	size_t _numL3Tasks;
	size_t _numNUMATasks;
	size_t _numMainTasks;
	size_t _stolenTasks;

	static const int MAX_L2_TASKS = 8;
	static const int MAX_L3_TASKS = 255;
public:
	ReadyQueueLocality(SchedulingPolicy policy, size_t numL2Queues, size_t numL3Queues, uint8_t numNUMAQueues)
		: ReadyQueue(policy),
		_numL2Queues(numL2Queues), _numL3Queues(numL3Queues), _numNUMAQueues(numNUMAQueues), _numReadyTasks(0)
	{
		_numL2Tasks = 0;
		_numL3Tasks = 0;
		_numNUMATasks = 0;
		_numMainTasks = 0;
		_stolenTasks = 0;

		_L2Queues = (ready_queue_t *) MemoryAllocator::alloc(numL2Queues * sizeof(ready_queue_t));
		_L3Queues = (ready_queue_t *) MemoryAllocator::alloc(numL3Queues * sizeof(ready_queue_t));
		_NUMAQueues = (main_ready_queue_t *) MemoryAllocator::alloc(numNUMAQueues * sizeof(main_ready_queue_t));

		for (size_t i = 0; i < _numL2Queues; i++) {
			new (&_L2Queues[i]) ready_queue_t(MAX_L2_TASKS);
		}

		for (size_t i = 0; i < _numL3Queues; i++) {
			new (&_L3Queues[i]) ready_queue_t(MAX_L3_TASKS);
		}

		for (uint8_t i = 0; i < _numNUMAQueues; i++) {
			new (&_NUMAQueues[i]) main_ready_queue_t();
		}
	}

	~ReadyQueueLocality()
	{
		for (size_t i = 0; i < _numL2Queues; i++) {
			_L2Queues[i].~ready_queue_t();
		}
		MemoryAllocator::free(_L2Queues, _numL2Queues * sizeof(ready_queue_t));

		for (size_t i = 0; i < _numL3Queues; i++) {
			_L3Queues[i].~ready_queue_t();
		}
		MemoryAllocator::free(_L3Queues, _numL3Queues * sizeof(ready_queue_t));

		for (uint8_t i = 0; i < _numNUMAQueues; i++) {
			_NUMAQueues[i].~main_ready_queue_t();
		}
		MemoryAllocator::free(_NUMAQueues, _numNUMAQueues * sizeof(main_ready_queue_t));

		assert(_mainQueue.empty());
		if (DataTrackingSupport::isTrackingReportEnabled()) {
			std::cout << "L2: " << _numL2Tasks << std::endl;
			std::cout << "L3: " << _numL3Tasks << std::endl;
			std::cout << "NUMA: " << _numNUMATasks << std::endl;
			std::cout << "Main: " << _numMainTasks << std::endl;
			std::cout << "stolen: " << _stolenTasks << std::endl;

			// Report caches miss rate
			std::cout << "L2 accesses: " << L2Cache::getAccessedBytes() << std::endl;
			std::cout << "L2 misses: " << L2Cache::getMissedBytes() << std::endl;
			std::cout << "L3 accesses: " << L3Cache::getAccessedBytes() << std::endl;
			std::cout << "L3 misses: " << L3Cache::getMissedBytes() << std::endl;
			//std::cout << "L2 miss rate: " << DataTrackingSupport::getMissRateL2() << std::endl;
			//std::cout << "L3 miss rate: " << DataTrackingSupport::getMissRateL3() << std::endl;
		}
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		unsigned int L2id = task->getL2hint();
		unsigned int L3id = task->getL3hint();
		uint8_t NUMAid = task->getNUMAhint();

		bool inserted = false;
		if (L2id != (unsigned int) -1) {
			assert(L2id < _numL2Queues);
			if (unblocked || _policy == LIFO_POLICY) {
				inserted = _L2Queues[L2id].push_front(task);
			} else {
				inserted = _L2Queues[L2id].push_back(task);
			}
			_numL2Tasks++;
		}

		if (!inserted && L3id != (unsigned int) -1) {
			assert(L3id < _numL3Queues);
			if (unblocked || _policy == LIFO_POLICY) {
				inserted = _L3Queues[L3id].push_front(task);
			} else {
				inserted = _L3Queues[L3id].push_back(task);
			}
			_numL3Tasks++;
		}

		if (!inserted && NUMAid != (uint8_t) -1) {
			assert(NUMAid < _numNUMAQueues);
			if (unblocked || _policy == LIFO_POLICY) {
				_NUMAQueues[NUMAid].push_front(task);
			} else {
				_NUMAQueues[NUMAid].push_back(task);
			}
			_numNUMATasks++;
			inserted = true;
		}

		if (!inserted) {
			if (unblocked || _policy == LIFO_POLICY) {
				_mainQueue.push_front(task);
			} else {
				_mainQueue.push_back(task);
			}
			_numMainTasks++;
			inserted = true;
		}

		++_numReadyTasks;
		FatalErrorHandler::failIf(!inserted, "Must have been inserted.");
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}


		// 1. Try to get high affinity task from L2 local queue
		unsigned int L2id = ((CPU *)computePlace)->getL2CacheId();
		if (!_L2Queues[L2id].empty()) {
			ready_queue_t *queue = &_L2Queues[L2id];
			Task *result = queue->pop_front();
			assert(result != nullptr);

			--_numReadyTasks;
			return result;
		}

		// 2. Try to get medium affinity task from L3 local queue
		unsigned int L3id = ((CPU *)computePlace)->getL3CacheId();
		if (!_L3Queues[L3id].empty()) {
			ready_queue_t *queue = &_L3Queues[L3id];
			Task *result = queue->pop_front();
			assert(result != nullptr);

			--_numReadyTasks;
			return result;
		}

		// 3. Try to get low affinity task from NUMA queue
		unsigned int NUMAid = ((CPU *)computePlace)->getNumaNodeId();
		if (!_NUMAQueues[NUMAid].empty()) {
			main_ready_queue_t *queue = &_NUMAQueues[NUMAid];
			Task *result = queue->front();
			assert(result != nullptr);

			queue->pop_front();
			--_numReadyTasks;
			return result;
		}

		// 4. Try to get no affinity task from main queue
		if (!_mainQueue.empty()) {
			main_ready_queue_t *queue = &_mainQueue;
			Task *result = queue->front();
			assert(result != nullptr);

			queue->pop_front();
			--_numReadyTasks;
			return result;
		}

		bool stolen = false;
		ready_queue_t *queue = nullptr;
		// 4. Try to steal from other L2 local queues that share L3
		for (size_t i = 0; i < _numL2Queues; i++) {
			if (i != L2id) {
				L2Cache *l2cache = HardwareInfo::getL2Cache(i);
				if (l2cache->getAssociatedL3Id() == L3id) {
					queue = &_L2Queues[i];
					if (!queue->empty()) {
						stolen = true;
						break;
					}
				}
			}
		}

		// 5. Try to steal from other L3 local queues
		if (queue->empty()) {
			for (size_t i = 0; i < _numL3Queues; i++) {
				if (i != L3id) {
					queue = &_L3Queues[i];
					if (!queue->empty()) {
						stolen = true;
						break;
					}
				}
			}
		}

		// 6. Try to steal from any L2 local queues
		if (queue->empty()) {
			for (size_t i = 0; i < _numL2Queues; i++) {
				if (i != L2id) {
					queue = &_L2Queues[i];
					if (!queue->empty()) {
						stolen = true;
						break;
					}
				}
			}
		}

		if (!queue->empty()) {
			Task *result = queue->pop_back();
			assert(result != nullptr);

			--_numReadyTasks;

			if (stolen) {
				_stolenTasks++;
				// Unset hints from stolen tasks. Otherwise, in the check
				// expiration mechanism, they will be reenqueued.
				unsigned int &l2hint = result->getL2hint();
				unsigned int &l3hint = result->getL3hint();
				l2hint = (unsigned int) -1;
				l3hint = (unsigned int) -1;
			}
			return result;
		}
		else {
			// 7. Try to steal from other NUMA queues
			if (DataTrackingSupport::isNUMAStealingEnabled()) {
				for (uint8_t i = 0; i < _numNUMAQueues; i++) {
					if (i != NUMAid) {
						main_ready_queue_t *numaQueue = &_NUMAQueues[i];
						if (!numaQueue->empty()) {
							_stolenTasks++;

							Task *result = numaQueue->back();
							assert(result != nullptr);

							numaQueue->pop_back();
							--_numReadyTasks;

							// Unset hints from stolen tasks. Otherwise, in the check
							// expiration mechanism, they will be reenqueued.
							unsigned int &l2hint = result->getL2hint();
							unsigned int &l3hint = result->getL3hint();
							l2hint = (unsigned int) -1;
							l3hint = (unsigned int) -1;

							return result;
						}
					}
				}
			}
		}

		FatalErrorHandler::failIf(DataTrackingSupport::isNUMAStealingEnabled(), "There must be a ready task.");
		return nullptr;
	}

	inline size_t getNumReadyTasks() const
	{
		return _numReadyTasks;
	}

};


#endif // READY_QUEUE_LOCALITY_HPP
