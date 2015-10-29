#ifndef INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP


#include "../InstrumentThreadManagement.hpp"
#include "InstrumentProfile.hpp"


namespace Instrument {
	inline void createdThread(WorkerThread *thread)
	{
		Profile::createdThread(thread);
	}
	
	inline void threadWillSuspend(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
}


#endif // INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
