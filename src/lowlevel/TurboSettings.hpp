/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef TURBO_SETTINGS_HPP
#define TURBO_SETTINGS_HPP

#ifdef __SSE2__
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

#include <atomic>

#include "support/config/ConfigVariable.hpp"
#include <MemoryAllocator.hpp>


class TurboSettings {
public:
	static void initialize()
	{
#ifdef __SSE2__
		ConfigVariable<bool> turbo("turbo.enabled");

		if (turbo.getValue()) {
			// Enable flush-to-zero (FZ) and denormals are zero (FAZ) floating-point
			// optimizations if available. All threads created by the runtime system
			// will inherit these optimization flags
			_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

			_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		}
#endif
	}

	static void shutdown()
	{
	}

	static bool isWarmupEnabled()
	{
		static ConfigVariable<bool> warmup("turbo.warmup");
		return warmup.getValue();
	}

	static void warmupPhase()
	{
		void *ptr = MemoryAllocator::alloc(4096);
		MemoryAllocator::free(ptr, 4096);
	}
};

#endif // TURBO_SETTINGS_HPP
