#define _GNU_SOURCE
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
	ctf_thread_id_t threadId = currentExternalThread->getInstrumentationId().threadId;
	contextUnbounded->writeContext(buf, threadId);
}
