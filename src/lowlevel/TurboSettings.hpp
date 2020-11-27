/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TURBO_SETTINGS_HPP
#define TURBO_SETTINGS_HPP

#ifdef __SSE2__
#include <xmmintrin.h>
#endif

#include "support/config/ConfigVariable.hpp"


class TurboSettings {
public:
	static void initialize()
	{
#ifdef __SSE2__
		ConfigVariable<bool> turbo("turbo.enabled");

		if (turbo.getValue()) {
			// Enable flush-to-zero (FZ) and denormals are zero (FAZ) floating-point optimizations
			// All threads created by the runtime system will inherit these optimization flags
			const unsigned int mask = 0x8080;
			unsigned int mxcsr = _mm_getcsr() | mask;
			_mm_setcsr(mxcsr);
		}
#endif
	}

	static void shutdown()
	{
	}
};

#endif // TURBO_SETTINGS_HPP
