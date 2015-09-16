#ifndef INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
#define INSTRUMENT_GRAPH_TASK_EXECUTION_HPP


#include "../InstrumentTaskExecution.hpp"
#include <InstrumentTaskId.hpp>


namespace Instrument {
	void startTask(task_id_t taskId, CPU *cpu);
	void returnToTask(task_id_t taskId, CPU *cpu);
	void endTask(task_id_t taskId, CPU *cpu);
}


#endif // INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
