#ifndef INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP


#include "../InstrumentThreadManagement.hpp"


namespace Instrument {
	inline void createdThread(__attribute__((unused)) WorkerThread *thread)
	{
	}
	
	inline void threadWillSuspend(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
}


#endif // INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
