/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include <MemoryAllocator.hpp>

#include "CTFStreamKernel.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

std::vector<ctf_kernel_event_size_t> *CTFAPI::CTFStreamKernel::_eventSizes = nullptr;

struct CTFAPI::CTFStreamKernel::KernelEventHeader *CTFAPI::CTFStreamKernel::mapStream()
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

	// TODO can I close the fd here?
	//rc = close(fd);
	//if (rc == -1) {
	//	FatalErrorHandler::warn("CTF: Kernel: When obtaining stream file size: ", strerror(errno));
	//}

	return (struct KernelEventHeader *) (_streamMap + sizeof(PacketHeader) + sizeof(PacketContext));
}

void CTFAPI::CTFStreamKernel::unmapStream()
{
	int rc = munmap(_streamMap, _streamSize);
	if (rc == 1) {
		FatalErrorHandler::warn(" when unmapping stream file");
	}
}

uint64_t CTFAPI::CTFStreamKernel::getEventSize(
	struct KernelEventHeader *current
) {
	return sizeof(struct KernelEventHeader) + (*_eventSizes)[current->id];
}

struct CTFAPI::CTFStreamKernel::KernelEventHeader *CTFAPI::CTFStreamKernel::getNextEvent(
	struct KernelEventHeader *current
) {
	return (struct KernelEventHeader *) (((char *) current) + getEventSize(current));
}

struct CTFAPI::CTFStreamKernel::KernelEventHeader *CTFAPI::CTFStreamKernel::getPreviousEvent(
	struct KernelEventHeader *current,
	struct Node *node
) {
	return (node->offset == 0)? nullptr :
		(struct KernelEventHeader *) (((char *)current) - (sizeof(struct KernelEventHeader) + node->offset));
}

// hole points to the eventList location where the current element should be
// stored. However, the current element is out-of-order and we need to move it
// backward. Hence, the first thing to do is to move the previous element into
// the hole. Then, we will continue moving the list and keeping track of how
// many elements (and their size) we need to move in the mapping. Finally, we
// will move all data with a memcopy.
void CTFAPI::CTFStreamKernel::moveUnsortedEvent(
	struct Node *eventList,
	char *swapArea,
	uint64_t hole,
	struct KernelEventHeader **current,
	struct KernelEventHeader *previous,
	uint64_t *currentSize,
	uint64_t previousSize
) {
	char *src, *dst;
	uint64_t size = previousSize;
	struct KernelEventHeader *shadow;
	const struct KernelEventHeader *initialPrevious = previous;

	// TODO remove me after debugging
	//uint64_t currentId = hole;

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
	*current = (struct KernelEventHeader *) (((char *) initialPrevious) + *currentSize);
	*currentSize = previousSize;

	// TODO remove me
	// Let's now iterate and see what we have done the first element printed
	// is the one that we have not moved "previous" and the last one is the
	// new "current"
	//std::cout << " ====== moved =====" << std::endl;
	//struct KernelEventHeader * iter = previous;
	//uint64_t j = hole - 1;
	//while (iter <= *current) {
	//	std::cout << "     - " << j << " with id " << iter->id << " timestamp: " << iter->timestamp << " iter: " << iter << " size: " << getEventSize(iter) << " payload: " << (*_eventSizes)[iter->id] << " offset: " << eventList[j].offset << std::endl;
	//	j++;
	//	iter = getNextEvent(iter);
	//}
	//std::cout << " ==================" << std::endl;
}

void CTFAPI::CTFStreamKernel::sortEvents()
{
	struct KernelEventHeader *current, *previous;
	uint64_t currentSize, previousSize;
	void *end;

	// This stream will be sorted only if it was detected an out-of-order event
	uint64_t numberOfUnorderedEvents = _kernelEventsProvider.getNumberOfUnorderedEvents();
	if (numberOfUnorderedEvents == 0)
		return;

	assert(_eventSizes != nullptr);

	// Build a list to keep track of events. This is needed obtan the
	// previous event from the current one.

	uint64_t numberOfEvents = _kernelEventsProvider.getNumberOfProcessedEvents();
	assert(numberOfEvents >= 2);
	//uint64_t numberOfEvents = 100000;

	uint64_t eventListSize = numberOfEvents * sizeof(struct Node);
	struct Node *eventList = (struct Node *) MemoryAllocator::alloc(eventListSize);

	std::cout << "CTF Kernel Stream on CPU " << _cpuId << " needs sorting. Please, wait... " << std::flush;

	// Allocate temporal memory to swap event positions. An event payload
	// might expand up to the sizeo of an uint16_t as limited by the Linux
	// kernel perf interface. We add an extra page to also fit the header in
	// the worse case scenario (paranoid)
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
	// TODO stop iterating once all detected unordered events are fixed
	uint64_t hole = 1;
	while (current < end) {

		assert(hole < numberOfEvents);

		// TODO add while check. iterate with a for instead?
		//if (hole >= numberOfEvents) {
		//	FatalErrorHandler::fail("CTF: Kernel: Sort: Attempt to iterate out of bounds");
		//}

		//std::cout << " - now processing event " << hole << " with id " << current->id << "/" << numberOfEvents << " current: " << current << " size: " << currentSize << " payload: " << (*_eventSizes)[current->id] << std::endl;
		if (current->timestamp >= previous->timestamp) {
			// The event is in the right order.  We store the size
			// of the previous event into the offset of the
			// current one which is used to move from this object to
			// the previous one once we lose track of the previous
			// event header location
			eventList[hole].offset = (uint16_t) (previousSize - sizeof(struct KernelEventHeader));
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
			//std::cout << "   - New current id: " << current->id << " at: " << current << std::endl;
		}

		// TODO remove me
		//if (hole >= 2) {
		//	struct KernelEventHeader *aux = current;
		//	//std::cout << "   - starting 2back & 2forw" << std::endl;
		//	current = getPreviousEvent(current, &eventList[hole]);
		//	//std::cout << "   - going back 1 id: " << current->id << " at: " << current << std::endl;
		//	current = getPreviousEvent(current, &eventList[hole-1]);
		//	//std::cout << "   - going back 2 id: " << current->id << " at: " << current << std::endl;
		//	current = getNextEvent(current);
		//	//std::cout << "   - going forw 1 id: " << current->id << " at: " << current << std::endl;
		//	current = getNextEvent(current);
		//	//std::cout << "   - going forw 2 id: " << current->id << " at: " << current << std::endl;
		//	if (current != aux) {
		//		std::cout << "SOMETHING went wrong! look ma'!" << std::endl;
		//		std::cout << "current: " << current << std::endl;
		//		std::cout << "aux    : " << aux << std::endl;
		//		FatalErrorHandler::fail("CTF: Kernel: Sort: Consistency fix failed");
		//	}
		//}

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
