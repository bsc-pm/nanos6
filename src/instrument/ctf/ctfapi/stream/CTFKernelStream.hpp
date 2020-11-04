/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef CTF_KERNEL_STREAM_HPP
#define CTF_KERNEL_STREAM_HPP

#include <cstddef>
#include <vector>

#include "CTFKernelEventsProvider.hpp"
#include "CTFStream.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


namespace CTFAPI {

	class CTFKernelStream : public CTFStream
	{
	private:

		struct Node {
			uint16_t offset;
		};

		// TODO I should unify this with the definition of CTFKerenelEventProvider
		struct __attribute__((__packed__)) KernelEventHeader {
			ctf_kernel_event_id_t id;
			uint64_t timestamp;
		};

		static std::vector<ctf_kernel_event_size_t> *_eventSizes;

		size_t _streamSize;
		char * _streamMap;
		CTFKernelEventsProvider _kernelEventsProvider;

		struct KernelEventHeader *mapStream();

		void unmapStream();

		uint64_t getEventSize(struct KernelEventHeader *current);

		struct KernelEventHeader *getNextEvent(struct KernelEventHeader *current);

		struct KernelEventHeader *getPreviousEvent(
			struct KernelEventHeader *current,
			struct Node *node
		);

		void moveUnsortedEvent(
			struct Node *eventList,
			char *swapArea,
			uint64_t hole,
			struct KernelEventHeader **current,
			struct KernelEventHeader *previous,
			uint64_t *currentSize,
			uint64_t previousSize
		);

	public:
		CTFKernelStream(size_t userSize, size_t kernelSize, ctf_cpu_id_t cpu, int node, std::string path)
			: CTFStream(userSize, cpu, node, path, CTFKernelStreamId),
			  _kernelEventsProvider(cpu, kernelSize)
		{
		}

		~CTFKernelStream()
		{
		}

		// TODO move back to private
		void sortEvents();

		static uint64_t minimumKernelVersion()
		{
			return CTFKernelEventsProvider::minimumKernelVersion();
		}

		static void setReferenceTimestamp(uint64_t timestamp)
		{
			CTFKernelEventsProvider::setReferenceTimestamp(timestamp);
		}

		static void setEvents(
			std::vector<ctf_kernel_event_id_t>   &enabledEvents,
			std::vector<ctf_kernel_event_size_t> &eventSizes
		) {
			_eventSizes = &eventSizes;
			CTFKernelEventsProvider::setEvents(enabledEvents, eventSizes);
		}

		void updateAvailableKernelEvents()
		{
			_kernelEventsProvider.updateHead();
		}

		void consumeKernelEvents()
		{
			_kernelEventsProvider.updateTail();
		}

		void enableKernelEvents()
		{
			_kernelEventsProvider.enable();
		}

		void disableKernelEvents()
		{
			_kernelEventsProvider.disable();
		}

		uint64_t getLostEventsCount()
		{
			return _kernelEventsProvider.getLostEventsCount();
		}

		void integrityCheck()
		{
			uint64_t throttle, unthrottle;
			throttle = _kernelEventsProvider.getThrottleEventsCount();
			unthrottle = _kernelEventsProvider.getUnthrottleEventsCount();

			if (throttle != 0)
				FatalErrorHandler::warn("CTF Kernel Stream reported ", throttle, " throttle perf events on core ", _cpuId, ". You might need to enable less events.");

			if (unthrottle != 0)
				FatalErrorHandler::warn("CTF Kernel Stream reported ", unthrottle, " unthrottle perf events on core ", _cpuId, ".");
		}

		bool readKernelEvents()
		{
			const uint64_t defaultMinSize = _kernelEventsProvider.getHeaderEventSize();
			uint64_t minSize;
			uint64_t read;
			uint64_t size;
			void *buf;

			minSize = defaultMinSize;
			while (_kernelEventsProvider.hasEvents()) {
				size = _circularBuffer.allocAtLeast(minSize);
				if (size == 0) {
					// The buffer is full, abort read
					return false;
				}
				buf = _circularBuffer.getBuffer();
				if (_kernelEventsProvider.read(buf, size, &read)) {
					// Some events have been read. At this
					// point there might be more events to
					// be read but not enough memory. Commit
					// what has been written so far and try
					// again.
					assert(read > 0);
					_circularBuffer.submit(read);
					minSize = defaultMinSize;
				} else if (read > 0) {
					// It was __not__ possible to read a single
					// event because the obtained chunk is too
					// small. Increase the minimum size and
					// try again.
					minSize = read;
				} else {
					// There were events, but none of them
					// were tracepoints. All events have
					// been processed. We can exit safely.
					break;
				}
			}

			return true;
		}

		inline void shutdown() override
		{
			_circularBuffer.shutdown();
			integrityCheck();
			sortEvents();
		}
	};

}

#endif // CTF_KERNEL_STREAM_HPP
