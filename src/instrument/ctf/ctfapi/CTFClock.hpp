/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_CLOCK_HPP
#define CTF_CLOCK_HPP

#include <time.h>

// We prefer CLOCK_MONOTONIC_RAW to prevent dynamic NTF time adjustments.
// However, if the system does not support it, we fall back to CLOCK_MONOTONIC

#ifdef CLOCK_MONOTONIC_RAW
#define CTF_CLOCK CLOCK_MONOTONIC_RAW
#else
#define CTF_CLOCK CLOCK_MONOTONIC
#endif

#endif // CTF_CLOCK_HPP
