/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_SUPPORT_HPP
#define SCHEDULER_SUPPORT_HPP

#include <cstdint>

class Task;

namespace SchedulerSupport {

	constexpr uint64_t roundup(const uint64_t x, const uint64_t y)
	{
		return (((x + (y - 1ULL)) / y) * y);
	}

	inline uint64_t roundToNextPowOf2(const uint64_t x)
	{
		return roundup(x, 1ULL << (63 - __builtin_clzll(x)));
	}

	inline bool isPowOf2(const uint64_t x)
	{
		return (__builtin_popcountll(x) == 1);
	}

	struct LocalityQueue {
		// This data structure must have size < 256 (8 bits).
		Task **_tasks;
		uint8_t _size;
		uint8_t _front;
		uint8_t _back;
		bool _full;

		LocalityQueue(uint8_t size)
			: _size(size), _front(0), _back(0), _full(false)
		{
			_tasks = (Task **) MemoryAllocator::alloc(_size * sizeof(Task *));
			for (uint8_t i = 0; i < _size; i++) {
				_tasks[i] = nullptr;
			}
		}

		~LocalityQueue()
		{
			assert(empty());
#ifndef NDEBUG
			for (uint8_t i = 0; i < _size; i++) {
				assert(_tasks[i] == nullptr);
			}
#endif
			MemoryAllocator::free(_tasks, _size * sizeof(Task *));
		}

		bool empty() const
		{
			//if head and tail are equal, we are empty
			return (!_full && (_front == _back));
		}

		bool full() const
		{
			//If tail is ahead the head by 1, we are full
			return _full;
		}

		size_t size() const
		{
			size_t size = _size;

			if(!_full)
			{
				if(_front >= _back)
				{
					size = _front - _back;
				}
				else
				{
					size = _size + _front - _back;
				}
			}

			return size;
		}

		bool push_back(Task *task)
		{
			if (full())
				return false;

			assert(task != nullptr);
			assert(_tasks[_back] == nullptr);

			_tasks[_back] = task;
			_back = (_back+1) % _size;
			assert(_back < _size);

			_full = (_front == _back);

			return true;
		}

		bool push_front(Task *task)
		{
			if (full())
				return false;

			assert(task != nullptr);

			_front = (uint8_t) (_front-1) % _size;
			assert(_front < _size);
			assert(_tasks[_front] == nullptr);
			_tasks[_front] = task;

			_full = (_front == _back);

			return true;
		}

		Task *pop_front()
		{
			if (empty())
				return nullptr;

			Task *result = _tasks[_front];
			assert(result != nullptr);

			_tasks[_front] = nullptr;
			_front = (_front+1) % _size;
			assert(_front < _size);

			_full = false;

			return result;
		}

		Task *pop_back()
		{
			if (empty())
				return nullptr;

			_back = (uint8_t) (_back-1) % _size;
			assert(_back < _size);

			Task *result = _tasks[_back];
			assert(result != nullptr);

			_tasks[_back] = nullptr;

			_full = false;

			return result;
		}
	};
}

#endif // SCHEDULER_SUPPORT_HPP
