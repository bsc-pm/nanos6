#ifndef INSTRUMENT_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_THREAD_MANAGEMENT_HPP


class WorkerThread;
class CPU;


namespace Instrument {
	void createdThread(WorkerThread *thread);
	void threadWillSuspend(WorkerThread *thread, CPU *cpu);
	void threadHasResumed(WorkerThread *thread, CPU *cpu);
}


#endif // INSTRUMENT_THREAD_MANAGEMENT_HPP
