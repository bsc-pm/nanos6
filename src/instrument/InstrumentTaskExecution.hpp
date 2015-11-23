#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentTaskId.hpp>


class Task;
class CPU;
class WorkerThread;


namespace Instrument {
	void startTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread);
	void returnToTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread);
	void endTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread);
	void destroyTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread);
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
