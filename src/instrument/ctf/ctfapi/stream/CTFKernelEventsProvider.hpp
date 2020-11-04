/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef CTF_KERNEL_EVENTS_PROVIDER
#define CTF_KERNEL_EVENTS_PROVIDER


#include <cstdint>
#include <string>
#include <vector>
#include <linux/perf_event.h>
#include <linux/version.h>

#include "../CTFTypes.hpp"

namespace CTFAPI {

	class CTFKernelEventsProvider {
		public:
			struct __attribute__((__packed__)) EventHeader {
				ctf_kernel_event_id_t id;
				uint64_t timestamp;
			};

			struct __attribute__((__packed__)) EventCommon {
				unsigned short common_type;
				unsigned char  common_flags;
				unsigned char  common_preempt_count;
				int            common_pid;
			};

			struct __attribute__((__packed__)) PerfRecordSample {
				struct perf_event_header header;
				uint64_t time;   // if PERF_SAMPLE_TIME
				uint32_t size;   // if PERF_SAMPLE_RAW
				char     data[]; // if PERF_SAMPLE_RAW
			};

			struct __attribute__((__packed__)) PerfLostRecord {
				struct perf_event_header header;
				uint64_t id;
				uint64_t lost;
			};

		private:
			static uint64_t _referenceTimestamp;

			const uint64_t _eventHeaderSize;

			uint64_t _metaSize;
			uint64_t _dataSize;
			uint64_t _totalSize;
			uint64_t _tempSize;
			uint64_t _dataMask;

			long long unsigned int _localHead;
			long long unsigned int _localTail;
			long long unsigned int _current;

			uint64_t _lost;
			uint64_t _throttle;
			uint64_t _unthrottle;
			uint64_t _numberOfEvents;
			uint64_t _numberOfUnorderedEvents;

			std::vector<int> _eventsFds;
			void *_kernelBuffer;
			void *_temporalBuffer;
			char *_dataRing;
			void *_metaPage;

			uint64_t _lastTimestamp;

			ctf_kernel_event_id_t _sched_switch_id;
			ctf_kernel_event_id_t _sched_wakeup_id;

			static std::vector<ctf_kernel_event_id_t>   *_enabledEvents;
			static std::vector<ctf_kernel_event_size_t> *_eventSizes;

			void *getNextEvent(uint64_t current);

			void initializeDebug();
			void checkPerfRecordSampleSize(PerfRecordSample *perfRecordSample);
			void checkPerfEventSize(PerfRecordSample *perfRecordSample, ctf_kernel_event_size_t expectedFieldsSize);
			void printKernelEvent(PerfRecordSample *perfRecordSample, ctf_kernel_event_size_t payloadSize);
			void warnUnknownPerfRecord(struct perf_event_header *header);

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
