/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef CTF_KERNEL_EVENTS_PROVIDER
#define CTF_KERNEL_EVENTS_PROVIDER

#include "linux/version.h"

#include <cstdint>
#include <string>
#include <vector>

#include "../CTFTypes.hpp"

namespace CTFAPI {

	class CTFKernelEventsProvider {
		private:
			static uint64_t _referenceTimestamp;

			uint64_t _lastTimestamp;
			long long unsigned int _lastCurrent;
			ctf_kernel_event_id_t _lastId;

			uint64_t _metaSize;
			uint64_t _dataSize;
			uint64_t _totalSize;
			uint64_t _tempSize;
			uint64_t _dataMask;

			const uint64_t _eventHeaderSize;
			uint64_t _lost;
			uint64_t _throttle;
			uint64_t _unthrottle;
			uint64_t _numberOfEvents;
			uint64_t _numberOfUnorderedEvents;

			long long unsigned int _localHead;
			long long unsigned int _localTail;
			long long unsigned int _current;

			std::vector<int> _eventsFds;
			void *_kernelBuffer;
			void *_temporalBuffer;
			char *_dataRing;
			void *_metaPage;

			static std::vector<ctf_kernel_event_id_t>   *_enabledEvents;
			static std::vector<ctf_kernel_event_size_t> *_eventSizes;

			void *getNextEvent(uint64_t current);
		public:
			CTFKernelEventsProvider(int cpu, size_t userSize);
			~CTFKernelEventsProvider();

			static uint64_t minimumKernelVersion()
			{
				return (uint64_t) KERNEL_VERSION(4, 1, 0 );
			}

			static void setReferenceTimestamp(uint64_t timestamp)
			{
				_referenceTimestamp = timestamp;
			}

			static void setEvents(
				std::vector<ctf_kernel_event_id_t>   &enabledEvents,
				std::vector<ctf_kernel_event_size_t> &eventSizes
			) {
				_enabledEvents = &enabledEvents;
				_eventSizes = &eventSizes;
			}

			inline uint64_t getHeaderEventSize() const
			{
				return _eventHeaderSize;
			}

			inline uint64_t getLostEventsCount() const
			{
				return _lost;
			}

			inline uint64_t getThrottleEventsCount() const
			{
				return _throttle;
			}

			inline uint64_t getUnthrottleEventsCount() const
			{
				return _unthrottle;
			}

			inline uint64_t getNumberOfUnorderedEvents() const
			{
				return _numberOfUnorderedEvents;
			}

			inline uint64_t getNumberOfProcessedEvents() const
			{
				return _numberOfEvents;
			}

			bool read(void *buf, uint64_t size, uint64_t *read);
			void updateHead();
			bool hasEvents();
			void updateTail();
			void enable();
			void disable();
	};

}

#endif // CTF_KERNEL_EVENTS_PROVIDER
