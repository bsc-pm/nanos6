#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include "CPU.hpp"
#include "DependencyDomain.hpp"
#include "WorkerThreadBase.hpp"
#include "performance/HardwareCountersThreadLocalData.hpp"
#include <InstrumentThreadLocalData.hpp>

#include <atomic>


class Task;
class ThreadManager;


class WorkerThread : public WorkerThreadBase {
private:
	//! Indicates that it is time for this thread to participate in the shutdown process
	std::atomic<bool> _mustShutDown;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;
	
	HardwareCountersThreadLocalData _hardwareCounters;
	
	Instrument::ThreadLocalData _instrumentationData;
	
	void initialize();
	void handleTask();
	
	//! Only the thread manager is supposed to call the suspend and resume methods. Any other use must go through it.
	friend class ThreadManager;
	
	
public:
	WorkerThread() = delete;
	
	inline WorkerThread(CPU * cpu);
	
	inline virtual ~WorkerThread();
	
	//! \brief get the currently assigned task to this thread
	inline Task *getTask();
	
	//! \brief set the task that this thread must run when it is resumed
	//!
	//! \param[in] task the task that the thread will run when it is resumed
	inline void setTask(Task *task);
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain const *getDependencyDomain() const;
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain *getDependencyDomain();
	
	inline HardwareCountersThreadLocalData &getHardwareCounters();
	
	inline Instrument::ThreadLocalData &getInstrumentationData();
	
	//! \brief handle a task
	//! This method is here to cover the case in which a task is run within the execution of another in the same thread
	inline void handleTask(Task *task);
	
	//! \brief code that the thread executes
	virtual void body();
	
	
	//! \brief turn on the flag to start the shutdown process
	inline void signalShutdown();
	
	//! \brief get the thread shutdown flag
	inline bool hasPendingShutdown();
	
	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThread *getCurrentWorkerThread();
	
};



#ifndef NDEBUG
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
#include "WorkerThreadImplementation.hpp"


#endif // WORKER_THREAD_HPP
