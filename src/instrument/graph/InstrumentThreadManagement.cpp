#include "../InstrumentThreadManagement.hpp"
#include "InstrumentGraph.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/SpinLock.hpp"
#include "tasks/Task.hpp"

#include <cassert>
#include <mutex>


namespace Instrument {
	using namespace Graph;
	
	
	void createdThread(WorkerThread *thread)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(thread) == _threadToId.end());
		_threadToId[thread] = _nextThreadId++;
	}
	
	void threadWillSuspend(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
	void threadHasResumed(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
	}
	
}

