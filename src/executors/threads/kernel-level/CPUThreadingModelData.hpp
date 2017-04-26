#ifndef CPU_THREADING_MODEL_DATA_HPP
#define CPU_THREADING_MODEL_DATA_HPP


#include <atomic>
#include <deque>


struct CPU;
class WorkerThread;


struct CPUThreadingModelData {
private:
	//! \brief a thread responsible for shutting down the rest of the threads and itself
	std::atomic<WorkerThread *> _shutdownControllerThread;
	
	//! \brief number of threads that must be shut down
	static std::atomic<long> _shutdownThreads;
	
	//! \brief last thread that joins any other thread
	static std::atomic<WorkerThread *> _mainShutdownControllerThread;
	
	friend class WorkerThreadBase;
	
public:
	CPUThreadingModelData()
		: _shutdownControllerThread(nullptr)
	{
	}
	
	void initialize(CPU *cpu);
	void shutdownPhase1(CPU *cpu);
	void shutdownPhase2(CPU *cpu);
};


#endif // CPU_THREADING_MODEL_DATA_HPP
