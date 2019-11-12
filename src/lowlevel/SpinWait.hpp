/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/


#ifndef SPIN_WAIT_HPP
#define SPIN_WAIT_HPP

#if defined(__powerpc__) || defined(__powerpc64__) || defined(__PPC__) || defined(__PPC64__) || defined(_ARCH_PPC) || defined(_ARCH_PPC64)
#define POWER9_ARCH
#endif

#if defined(__SSE2__)
#define KNL_ARCH
#endif

#if defined(__i386__) || defined(__x86_64__)
#define MN4_ARCH
#endif

#if defined(__arm__) || defined(__aarch64__)
#define ARM_ARCH
#endif

#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE 64
#endif

#ifdef KNL_ARCH
#include <xmmintrin.h>
#endif

#ifdef POWER9_ARCH
/* Macros for adjusting thread priority (hardware multi-threading) */
#define HMT_very_low()    asm volatile("or 31,31,31   # very low priority")
#define HMT_low()         asm volatile("or 1,1,1	  # low priority")
#define HMT_medium_low()  asm volatile("or 6,6,6	  # medium low priority")
#define HMT_medium()      asm volatile("or 2,2,2	  # medium priority")
#define HMT_medium_high() asm volatile("or 5,5,5	  # medium high priority")
#define HMT_high()        asm volatile("or 3,3,3	  # high priority")
#define HMT_barrier()     asm volatile("" : : : "memory")
#endif

	
static inline void spinWait()
{
#ifdef KNL_ARCH
	_mm_pause();
#elif defined(POWER9_ARCH)
	do { HMT_low(); HMT_medium(); HMT_barrier(); } while (0);
#elif defined(MN4_ARCH)
	asm volatile("pause" ::: "memory");
#elif defined(ARM_ARCH)
	__asm__ __volatile__ ("yield");
#else
	#pragma message ("No 'pause' instruction/intrisic found for this architecture ")
#endif
}

template<class T, size_t Size = CACHELINE_SIZE>
class Padded : public T {
	constexpr static size_t roundup(size_t const x, size_t const y) {
		return ((((x) + ((y) - 1)) / (y)) * (y));
	}
	uint8_t padding[roundup(sizeof(T), Size)-sizeof(T)];
};

#endif // SPIN_WAIT_HPP
