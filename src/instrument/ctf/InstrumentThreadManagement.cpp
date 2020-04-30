#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>

#include "ctfapi/CTFTypes.hpp"
#include "InstrumentThreadManagement.hpp"

ctf_thread_id_t Instrument::gettid(void)
{
	return syscall(SYS_gettid);
}
