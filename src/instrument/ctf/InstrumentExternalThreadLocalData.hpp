/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_EXTERNAL_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_CTF_EXTERNAL_THREAD_LOCAL_DATA_HPP


#include <atomic>
#include <cstdint>
#include <string>


namespace Instrument {
	struct ExternalThreadLocalData {
		static std::atomic<uint32_t> externalThreadCount;

		ExternalThreadLocalData(__attribute__((unused)) std::string const &externalThreadName)
		{
			// We keep the external threads count into a static
			// ExternalThreadLocalData instead of a CTF structure
			// such as CTFTrace because at this point the CTF
			// backend may have not been initialized yet
			externalThreadCount.fetch_add(1, std::memory_order_relaxed);
		}

		static uint32_t getExternalThreadCount() {
			return externalThreadCount.load(std::memory_order_relaxed);
		}
	};
}


#endif // INSTRUMENT_CTF_EXTERNAL_THREAD_LOCAL_DATA_HPP
