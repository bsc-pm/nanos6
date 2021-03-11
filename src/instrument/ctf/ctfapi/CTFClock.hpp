/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_CLOCK_HPP
#define CTF_CLOCK_HPP

#include <time.h>

// Always use the CLOCK_MONOTONIC clock as it is drift-corrected by NTP,
// and is not affected by time jumps.
#define CTF_CLOCK CLOCK_MONOTONIC

#endif // CTF_CLOCK_HPP
