/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>

#include <lowlevel/threads/ExternalThread.hpp>

#include "CTFStreamUnboundedPrivate.hpp"
#include "../CTFTypes.hpp"

void CTFAPI::CTFStreamUnboundedPrivate::writeContext(void **buf)
{
	assert(context != nullptr);
	CTFContextUnbounded *contextUnbounded = (CTFContextUnbounded *) context;
	ExternalThread *currentExternalThread = ExternalThread::getCurrentExternalThread();
	assert(currentExternalThread != nullptr);
	ctf_thread_id_t tid = currentExternalThread->getInstrumentationId().tid;
	contextUnbounded->writeContext(buf, tid);
}
