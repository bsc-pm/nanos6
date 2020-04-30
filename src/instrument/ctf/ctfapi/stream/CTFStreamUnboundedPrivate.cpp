#define _GNU_SOURCE
#include <sched.h>
#include <cassert>

#include "CTFStreamUnboundedPrivate.hpp"
#include "../CTFTypes.hpp"

void CTFAPI::CTFStreamUnboundedPrivate::writeContext(void **buf)
{
	assert(context != nullptr);
	CTFContextUnbounded *contextUnbounded = (CTFContextUnbounded *) context;
	ctf_cpu_id_t cpuId = sched_getcpu();
	contextUnbounded->writeContext(buf, cpuId);
}
