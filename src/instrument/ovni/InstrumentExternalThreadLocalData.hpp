/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_EXTERNAL_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_OVNI_EXTERNAL_THREAD_LOCAL_DATA_HPP


#include <atomic>
#include <cstdint>
#include <string>


namespace Instrument {
	struct ExternalThreadLocalData {
		static std::atomic<uint32_t> externalThreadCount;

		ExternalThreadLocalData(__attribute__((unused)) std::string const &externalThreadName)
		{
			externalThreadCount.fetch_add(1, std::memory_order_relaxed);
		}

		static uint32_t getExternalThreadCount() {
			return externalThreadCount.load(std::memory_order_relaxed);
		}
	};
}


#endif // INSTRUMENT_OVNI_EXTERNAL_THREAD_LOCAL_DATA_HPP
