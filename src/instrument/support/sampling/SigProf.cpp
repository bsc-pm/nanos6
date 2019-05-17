/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#if HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>


#include "SigProf.hpp"
#include "ThreadLocalData.hpp"

#include <lowlevel/FatalErrorHandler.hpp>

#include <atomic>


// Workaround for missing definition
#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif



namespace Instrument {
namespace Sampling {


bool SigProf::_enabled = false;
unsigned int SigProf::_nsPerSample = 1000;
SigProf::handler_t SigProf::_handler = nullptr;


void sigprofHandler(__attribute__((unused)) int signal, __attribute__((unused)) siginfo_t *sigInfo, __attribute__((unused)) void *signalContext)
{
	// NOTE: This line fails if Instrument::ThreadLocalData does not inherit from Instrument::Sampling::ThreadLocalData
	Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData();
	
	if (__builtin_expect(!SigProf::isEnabled(), 0)) {
		struct itimerspec it;
		it.it_interval = {0, 0};
		it.it_value = {0, 0};
		
		// Disarm the timer
		timer_settime(threadLocal._profilingTimer, 0, &it, 0);
		
		return;
	}
	
	if ((threadLocal._disableCount > 0) || (threadLocal._lightweightDisableCount > 0)) {
		// Temporarily disabled
		return;
	}
	
	if (SigProf::getHandler() != nullptr) {
		SigProf::getHandler()(threadLocal);
	}
}


void SigProf::init()
{
	#if !defined(HAVE_BACKTRACE) && !defined(HAVE_LIBUNWIND)
		std::cerr << "Warning: profiling currently not supported in this platform." << std::endl;
		return;
	#endif
	
	struct sigaction sa;
	sa.sa_sigaction = (void (*)(int, siginfo_t *, void *)) sigprofHandler;
	sigemptyset(&sa.sa_mask);
	sigaddset(&sa.sa_mask, SIGPROF);
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	
	int rc = sigaction(SIGPROF, &sa, 0);
	FatalErrorHandler::handle(rc, " programming the SIGPROF signal handler");
	
	_enabled = true;
	std::atomic_thread_fence(std::memory_order_seq_cst);
}


void SigProf::setUpThread(Instrument::Sampling::ThreadLocalData &threadLocal)
{
	#if !defined(HAVE_BACKTRACE) && !defined(HAVE_LIBUNWIND)
		std::cerr << "Warning: profiling currently not supported in this platform." << std::endl;
		return ;
	#endif
	
	struct sigevent se;
	se.sigev_notify = SIGEV_THREAD_ID;
	se.sigev_signo = SIGPROF;
	se.sigev_value.sival_int = 1;
	se.sigev_notify_thread_id = syscall(SYS_gettid);
	
	// Profiling actually starts after the follwoing lines
	int rc = timer_create(CLOCK_THREAD_CPUTIME_ID, &se, &threadLocal._profilingTimer);
	FatalErrorHandler::handle(rc, " creating a timer for profiling");
}


void SigProf::enableThread(Instrument::Sampling::ThreadLocalData &threadLocal)
{
	assert(threadLocal._disableCount > 0);
	
	threadLocal._disableCount--;
	if (threadLocal._disableCount > 0) {
		return;
	}
	
	struct itimerspec it = {
		.it_interval = { .tv_sec = 0, .tv_nsec = _nsPerSample },
		.it_value = { .tv_sec = 0, .tv_nsec = _nsPerSample }
	};
	
	int rc = timer_settime(threadLocal._profilingTimer, 0, &it, 0);
	FatalErrorHandler::handle(rc, " arming the timer for profiling");
}


void SigProf::disableThread(Instrument::Sampling::ThreadLocalData &threadLocal)
{
	assert(threadLocal._disableCount >= 0);
	
	threadLocal._disableCount++;
	if (threadLocal._disableCount > 1) {
		return;
	}
	
	struct itimerspec it = {
		.it_interval = { .tv_sec = 0, .tv_nsec = 0 },
		.it_value = { .tv_sec = 0, .tv_nsec = 0 }
	};
	
	int rc = timer_settime(threadLocal._profilingTimer, 0, &it, 0);
	FatalErrorHandler::handle(rc, " disarming the timer for profiling");
}


bool SigProf::lightweightEnableThread(Instrument::Sampling::ThreadLocalData &threadLocal)
{
	assert(threadLocal._lightweightDisableCount > 0);
	return threadLocal._lightweightDisableCount-- == 0;
}


bool SigProf::lightweightDisableThread(Instrument::Sampling::ThreadLocalData &threadLocal)
{
	assert(threadLocal._lightweightDisableCount >= 0);
	return threadLocal._lightweightDisableCount++ == 0;
}


void SigProf::forceHandler()
{
	sigprofHandler(0, nullptr, nullptr);
}


} // namespace Sampling
} // namespace Instrument

