/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/
/*
	Polling each register correctly depending on architectures is code
	obtained the FFTW library.
	
	Copyright (c) 2003, 2007-8 Matteo Frigo
	Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
	
	Permission is hereby granted, free of charge, to any person obtaining
	a copy of this software and associated documentation files (the
	"Software"), to deal in the Software without restriction, including
	without limitation the rights to use, copy, modify, merge, publish,
	distribute, sublicense, and/or sell copies of the Software, and to
	permit persons to whom the Software is furnished to do so, subject to
	the following conditions:
	
	The above copyright notice and this permission notice shall be
	included in all copies or substantial portions of the Software.
	
	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
	NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
	LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
	OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
	WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef CHRONO_ARCH_DEPENDENT_HPP
#define CHRONO_ARCH_DEPENDENT_HPP

#include <iostream>
#include <sys/time.h>

#include "nanos6/polling.h"


#if defined(__ARM_ARCH_7A__) && defined(ARMV7A_HAS_CNTVCT)
	// ARMV7 counter using the CNTVCT cycle counter
	// Activated through the --enable-armv7a-cntvct configure option
	typedef uint64_t ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		uint32_t rt1, rt2 = 0;
		asm volatile("mrrc p15, 1, %0, %1, c14" : "=r"(rt1), "=r"(rt2));
		
		return ((uint64_t)rt1) | (((uint64_t)rt2) << 32);
	}
#elif defined(__aarch64__) && defined(HAVE_ARMV8_CNTVCT_EL0) && !defined(HAVE_ARMV8CC)
	// ARMV8 counter using the CNTVCT_EL0 cycle counter
	// Activated through the --enable-armv8-cntvct-el0 configure option
	typedef uint64_t ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		uint64_t rt;
		asm volatile("mrs %0, CNTVCT_EL0" : "=r" (rt));
		
		return rt;
	}
#elif defined(__aarch64__) && defined(HAVE_ARMV8CC)
	// ARMV8 counter using the PMCCNTR_EL0 cycle counter
	// Activated through the --enable-armv8cyclecounter configure option
	typedef uint64_t ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		uint64_t cc = 0;
		asm volatile("mrs %0, PMCCNTR_EL0" : "=r"(cc));
		
		return cc;
	}
#elif ((((defined(__GNUC__) && (defined(__powerpc__) || defined(__ppc__))) || (defined(__MWERKS__) && defined(macintosh)))) || (defined(__IBM_GCC_ASM) && (defined(__powerpc__) || defined(__ppc__))))
	// PowerPC 'cycle' counter using the time base register
	typedef unsigned long long ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		unsigned int tbl, tbu0, tbu1;
		
		do {
			asm volatile ("mftbu %0" : "=r"(tbu0));
			asm volatile ("mftb %0"  : "=r"(tbl));
			asm volatile ("mftbu %0" : "=r"(tbu1));
		} while (tbu0 != tbu1);
		
		return (((unsigned long long)tbu0) << 32) | tbl;
	}
#elif (defined(__GNUC__) || defined(__ICC)) && defined(__i386__)
	// Pentium cycle counter
	typedef unsigned long long ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		ticks_t ret;
		asm volatile("rdtsc" : "=A" (ret));
		
		return ret;
	}
#elif (defined(__GNUC__) || defined(__ICC) || defined(__SUNPRO_C)) && defined(__x86_64__)
	// X86-64 cycle counter
	typedef unsigned long long ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		unsigned a, d;
		asm volatile ("rdtsc" : "=a" (a), "=d" (d));
		
		return ((ticks_t)a) | (((ticks_t)d) << 32);
	}
#elif defined(__GNUC__) && defined(__ia64__)
	// GCC
	typedef unsigned long ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		ticks_t ret;
		asm volatile ("mov %0=ar.itc" : "=r"(ret));
		
		return ret;
	}
#else
	// If everything else fails, use standard library clock ticks
	typedef std::clock ticks_t;
	
	static inline ticks_t getTicks(void)
	{
		return std::clock();
	}
#endif


namespace ChronoArchDependentData {
	extern double _tickConversionFactor;
}


class Chrono {

private:
	
	ticks_t _start;
	
	ticks_t _end;
	
	size_t _accumulated;
	
	
public:
	
	inline Chrono() :
		_accumulated(0)
	{
	}
	
	inline Chrono(size_t ticks) :
		_accumulated(ticks)
	{
	}
	
	
	inline void start()
	{
		_start = getTicks();
	}
	
	inline void stop()
	{
		_end = getTicks();
		_accumulated += (((double) _end) - ((double) _start));
	}
	
	inline void restart()
	{
		_accumulated = 0;
	}
	
	inline void continueAt(Chrono &other)
	{
		stop();
		other.start();
	}
	
	//! \brief Returns a representation in microseconds of the accumulated
	//! time gathered by the chronometer
	inline operator double() const
	{
		return ((double) _accumulated) * ChronoArchDependentData::_tickConversionFactor;
	}
	
	inline void operator+=(const Chrono &chrono)
	{
		_accumulated += chrono.getAccumulated();
	}
	
	//! \brief Returns the accumulated ticks of this chronometer, not
	//! converted to time
	inline size_t getAccumulated() const
	{
		return _accumulated;
	}
	
};


class TickConversionUpdater {

private:
	
	//! Structs to compare ticks with execution time
	timeval _t1, _t2;
	
	//! A chrono measuring clock ticks
	Chrono _c1;
	
	//! Whether an update cycle has started
	bool _updateStarted;
	
	//! A stopwatch to know when it is needed to update tick conversion factor
	Chrono _updateFrequencyChrono;
	
	//! The singleton instance
	static TickConversionUpdater *_tickUpdater;
	
	
private:
	
	inline TickConversionUpdater() :
		_c1(),
		_updateStarted(false),
		_updateFrequencyChrono()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	TickConversionUpdater(TickConversionUpdater const&) = delete;
	TickConversionUpdater(TickConversionUpdater&&) = delete;
	TickConversionUpdater& operator=(TickConversionUpdater const&) = delete;
	TickConversionUpdater& operator=(TickConversionUpdater &&) = delete;
	
	
	//! \brief Initialize the tick conversion updater
	static void initialize();
	
	//! \brief Finalize the tick conversion updater
	static void shutdown();
	
	//! \brief Begin the update of the conversion factor computation
	static void beginUpdate();
	
	//! \brief Finish the update of the conversion factor computation
	static void finishUpdate();
	
	//! \brief Service routine that updates the tick conversion factor
	//!
	//! \return Whether the service routine should be stopped
	static int updateTickConversionFactor(void *);
	
};

#endif // CHRONO_ARCH_DEPENDENT_HPP
