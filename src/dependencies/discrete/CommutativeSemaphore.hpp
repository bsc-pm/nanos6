/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMMUTATIVE_SEMAPHORE_HPP
#define COMMUTATIVE_SEMAPHORE_HPP

#include <cstdint>
#include <list>
#include <bitset>

#include "lowlevel/PaddedTicketSpinLock.hpp"

#include <config.h>

class Task;
struct CPUDependencyData;

constexpr int commutative_mask_bits = CACHELINE_SIZE * 8;

struct CommutativeSemaphore {
	typedef std::bitset<commutative_mask_bits> commutative_mask_t;
	typedef PaddedTicketSpinLock<> lock_t;
	typedef std::list<Task *> queue_t;

	static lock_t _lock;
	static commutative_mask_t _mask;
	static queue_t _queue;

	static bool registerTask(Task *task);
	static void releaseTask(Task *task, CPUDependencyData &hpDependencyData);

	static inline commutative_mask_t getMaskForAddress(void *address)
	{
		commutative_mask_t mask = 1;
		return (mask << addressHash(address));
	}

private:
	static inline int addressHash(void *address)
	{
		unsigned long long hash = 5381;
		unsigned long long uintAddress = (unsigned long long)address;

		hash = ((hash << 5) + hash) + ((char)uintAddress & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 8) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 16) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 24) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 32) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 40) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 48) & 0xFF);
		hash = ((hash << 5) + hash) + ((char)(uintAddress >> 56) & 0xFF);

		return (hash % commutative_mask_bits);
	}
};

#endif // COMMUTATIVE_SEMAPHORE_HPP