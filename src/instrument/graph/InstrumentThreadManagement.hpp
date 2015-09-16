#ifndef INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP


#include "../InstrumentThreadManagement.hpp"


namespace Instrument {
	void createdThread(WorkerThread *thread);
	void threadWillSuspend(WorkerThread *thread, CPU *cpu);
	void threadHasResumed(WorkerThread *thread, CPU *cpu);
	
}


#endif // INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP
