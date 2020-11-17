/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "CTFKernelStream.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <MemoryAllocator.hpp>

std::vector<ctf_kernel_event_size_t> *CTFAPI::CTFKernelStream::_eventSizes = nullptr;

CTFAPI::CTFKernelEventsProvider::EventHeader *CTFAPI::CTFKernelStream::mapStream()
{
	int fd, rc;
	struct stat stat;

	fd = open(_path.c_str(), O_RDWR);
	if (fd == -1) {
		FatalErrorHandler::fail("CTF: Kernel: When opening a stream file: ", strerror(errno));
	}

	rc = fstat(fd, &stat);
	if (rc == -1) {
		FatalErrorHandler::fail("CTF: Kerenl: When obtaining stream file size: ", strerror(errno));
	}
	_streamSize = stat.st_size;

	_streamMap = (char *) mmap(NULL, _streamSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (_streamMap == MAP_FAILED) {
		FatalErrorHandler::fail("CTF: Kernel: When mapping a stream file: ", strerror(errno));
	}

	rc = close(fd);
	if (rc == -1) {
		FatalErrorHandler::warn("CTF: Kernel: When closing stream file: ", strerror(errno));
	}

	return (CTFKernelEventsProvider::EventHeader *) (_streamMap + sizeof(PacketHeader) + sizeof(PacketContext));
}

void CTFAPI::CTFKernelStream::unmapStream()
{
	int rc = munmap(_streamMap, _streamSize);
	if (rc == 1) {
		FatalErrorHandler::warn(" when unmapping stream file");
	}
}

uint64_t CTFAPI::CTFKernelStream::getEventSize(
	CTFAPI::CTFKernelEventsProvider::EventHeader *current
) {
	return sizeof(CTFKernelEventsProvider::EventHeader) + (*_eventSizes)[current->id];
}

CTFAPI::CTFKernelEventsProvider::EventHeader *CTFAPI::CTFKernelStream::getNextEvent(
	CTFAPI::CTFKernelEventsProvider::EventHeader *current
) {
	return (CTFKernelEventsProvider::EventHeader *) (((char *) current) + getEventSize(current));
}

CTFAPI::CTFKernelEventsProvider::EventHeader *CTFAPI::CTFKernelStream::getPreviousEvent(
	CTFAPI::CTFKernelEventsProvider::EventHeader *current,
	struct Node *node
) {
	return (node->offset == 0)? nullptr :
		(CTFKernelEventsProvider::EventHeader *) (((char *)current) - (sizeof(CTFKernelEventsProvider::EventHeader) + node->offset));
}

// Hole points to the eventList location where the current element should be
// stored. However, the current element is out-of-order and we need to move it
// backward. Hence, the first thing to do is to move the previous element into
// the hole. Then, we will continue moving the list and keeping track of how
// many elements (and their size) we need to move in the mapping. Finally, we
// will move all data with a memcopy.
void CTFAPI::CTFKernelStream::moveUnsortedEvent(
	struct Node *eventList,
	char *swapArea,
	uint64_t hole,
	CTFAPI::CTFKernelEventsProvider::EventHeader **current,
	CTFAPI::CTFKernelEventsProvider::EventHeader *previous,
	uint64_t *currentSize,
	uint64_t previousSize
) {
	char *src, *dst;
	uint64_t size = previousSize;
	CTFKernelEventsProvider::EventHeader *shadow;
	const CTFKernelEventsProvider::EventHeader *initialPrevious = previous;

	// We need to move the first element unconditionally, otherwise we would
	// not be here right now
	eventList[hole] = eventList[hole - 1];
	hole--;

	// Position the intitial iterators. Previous points to the event we are
	// comparing "current". Shadow points to the event that was "previous"
	// in the previous iteration
	shadow = previous;
	previous = getPreviousEvent(previous, &eventList[hole]);

	while (previous != nullptr && (*current)->timestamp < previous->timestamp) {
		//std::cout << "  - going back one element! Now comparing id " << currentId << " against " << (hole-1) << " with id " << previous->id << " size: " << getEventSize(previous) << std::endl;
		// We also need to move this event, let's account it's size
		size += getEventSize(previous);
		// Move it's entry on the eventList
		eventList[hole] = eventList[hole - 1];
		hole--;
		// And get the new previous event, we keep the old value in
		// shadow for later use
		shadow = previous;
		previous = getPreviousEvent(previous, &eventList[hole]);
	}

	// We have found the spot where current needs to be moved!  First, we
	// update the event list. In the current "hole" we need to put the data
	// for "current". Which is basically the size of the event's payload
	// found before "current". This data is now stored in the event just
	// after us (because that event was in our position a few moments ago).
	// Therefore, let's update the "current" eventList entry with the info
	// of the event found after us.
	eventList[hole] = eventList[hole + 1];
	// We also need to modify the offset of the event next to the moved
	// "current", because it need to hold the payload size of "current"
	eventList[hole + 1].offset = (*_eventSizes)[(*current)->id];

	// Previous is in the right position, but we need to move everything
	// else past it and until current. The moved elements will overwrite
	// current, so we need to copy current in the swap area first.
	src = (char *) *current;
	dst = swapArea;
	std::memcpy(dst, src, *currentSize);
	// Then we move all the previous events forward. We use memmove because
	// memory areas might overlap
	src = (char *) shadow;
	dst = src + *currentSize;
	std::memmove(dst, src, size);
	// Then we copy back the current element on its new position
	src = swapArea;
	dst = (char *) shadow;
	std::memcpy(dst, src, *currentSize);
	// And finally we update the new "current". The "previous" element is
	// now in the eventList position of the old "current". But the exact
	// position might differ. We know that previous (and all other elements)
	// have been moved forward by "currentSize", so let's advance the
	// pointer.
	*current = (CTFKernelEventsProvider::EventHeader *) (((char *) initialPrevious) + *currentSize);
	*currentSize = previousSize;

	// Let's now iterate and see what we have done the first element printed
	// is the one that we have not moved "previous" and the last one is the
	// new "current"
	//std::cout << " ----- moved ------" << std::endl;
	//CTFKernelEventsProvider::EventHeader * iter = previous;
	//uint64_t j = hole - 1;
	//while (iter <= *current) {
	//	std::cout << "     - " << j << " with id " << iter->id << " timestamp: " << iter->timestamp << " iter: " << iter << " size: " << getEventSize(iter) << " payload: " << (*_eventSizes)[iter->id] << " offset: " << eventList[j].offset << std::endl;
	//	j++;
	//	iter = getNextEvent(iter);
	//}
	//std::cout << " ---------------- " << std::endl;
}

// This function iterates the events stored in a stream file and checks one by
// one whether its timestamp is greater than the previous event's timestamp. If
// it is not, it copies the unordered event into a "swap" memory area, and moves
// backward in the stream file until it finds the right position of the event
// (when its timestamp is bigger than the previous event).  Once the position of
// the unsorted event is found, all events between this position and the
// unsorted event position are physically moves forward and then it copies the
// unorered event in the created hole.
void CTFAPI::CTFKernelStream::sortEvents()
{
	CTFAPI::CTFKernelEventsProvider::EventHeader *current, *previous;
	uint64_t currentSize, previousSize;
	uint64_t fixedEvents = 0;
	void *end;

	// This stream will be sorted only if it was detected an out-of-order event
	uint64_t numberOfUnorderedEvents = _kernelEventsProvider.getNumberOfUnorderedEvents();
	if (numberOfUnorderedEvents == 0)
		return;

	assert(_eventSizes != nullptr);

	uint64_t numberOfEvents = _kernelEventsProvider.getNumberOfProcessedEvents();
	assert(numberOfEvents >= 2);

	// While iterating the stream file, we can walk the events forward but
	// not backwards. Therefore, we need to create a list which allows to
	// travers the events back. Here we are allocating such list.
	uint64_t eventListSize = numberOfEvents * sizeof(struct Node);
	struct Node *eventList = (struct Node *) MemoryAllocator::alloc(eventListSize);

	std::cout << "CTF Kernel Stream on CPU " << _cpuId << " needs sorting. Please, wait... " << std::flush;

	// Allocate temporal memory to swap event positions. An event payload
	// might expand up to the sizeo of an uint16_t as limited by the Linux
	// kernel perf interface. We add an extra page to also fit the header in
	// the worse case scenario (paranoid)
	// TODO should the swap area be static and shared?
	uint64_t eventSwapAreaSize = (1<<16) + (1<<12);
	char *swapArea = (char *) MemoryAllocator::alloc(eventSwapAreaSize);

	// Map the whole stream file into memory and get an index to the first event
	previous = mapStream();
	previousSize = getEventSize(previous);
	// Get the next event with respect to the current one
	current  = getNextEvent(previous);
	currentSize = getEventSize(current);
	// Calculate the end of the mapping
	end  = (void *) (_streamMap + _streamSize);
	// Initialize first event in the event list
	eventList[0].offset = 0;

	//std::cout << " - now processing event " << 0 << " with id " << previous->id << " current: " << previous << " size: " << previousSize << " (initial event)" << std::endl;

	// Iterate elements one by one, we start at the second element (index 1)
	uint64_t hole = 1;
	while ((current < end) && (fixedEvents != numberOfUnorderedEvents)) {

		assert(hole < numberOfEvents);

		//std::cout << " - now processing event " << hole << " with id " << current->id << "/" << numberOfEvents << " current: " << current << " size: " << currentSize << " payload: " << (*_eventSizes)[current->id] << std::endl;
		if (current->timestamp >= previous->timestamp) {
			// The event is in the right order.  We store the size
			// of the previous event into the offset of the
			// current one which is used to move from this object to
			// the previous one once we lose track of the previous
			// event header location
			eventList[hole].offset = (uint16_t) (previousSize - sizeof(CTFKernelEventsProvider::EventHeader));
		} else {
			// The current event is out-of-order. We need to move it
			// backwards. This will also update the "current" size
			// and the "current" pointer after moving. The
			// "previuos" pointer and size will no longer be
			// updated, though! But they are overwritten soon.
			//std::cout << " - Unordered event detected!" << std::endl;
			moveUnsortedEvent(eventList, swapArea, hole,
					  &current, previous,
					  &currentSize, previousSize);
			fixedEvents++;
		}

#ifndef NDEBUG
		if (hole >= 2) {
			CTFKernelEventsProvider::EventHeader *aux = current;
			current = getPreviousEvent(current, &eventList[hole]);
			current = getPreviousEvent(current, &eventList[hole-1]);
			current = getNextEvent(current);
			current = getNextEvent(current);
			if (current != aux) {
				FatalErrorHandler::fail("CTF: Kernel: Sort: Consistency fix failed");
			}
		}
#endif

		previous = current;
		current = getNextEvent(current);
		previousSize = currentSize;
		currentSize = getEventSize(current);
		hole++;
	}

	MemoryAllocator::free(swapArea, eventSwapAreaSize);
	MemoryAllocator::free(eventList, eventListSize);
	unmapStream();

	std::cout << "[DONE]" << std::endl << std::flush;
}
