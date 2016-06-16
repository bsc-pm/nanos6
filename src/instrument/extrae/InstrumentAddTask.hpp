#ifndef INSTRUMENT_EXTRAE_ADD_TASK_HPP
#define INSTRUMENT_EXTRAE_ADD_TASK_HPP


//#include "../InstrumentAddTask.hpp"
#include <InstrumentTaskId.hpp>
#include "InstrumentExtrae.hpp"

#include <cassert>
#include <mutex>


class Task;


namespace Instrument {
	inline task_id_t enterAddTask(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_CREATION;
		
		Extrae_emit_CombinedEvents ( &ce );
		
		assert(taskInfo != nullptr);
		assert(taskInvokationInfo != nullptr);
		const char * label = taskInfo->task_label? taskInfo->task_label: taskInvokationInfo->invocation_source;
		{
			std::lock_guard<SpinLock> guard(_userFunctionMapLock);
			_userFunctionMap[(void *) taskInfo->run] = label;
		}
		
		return task_id_t((void *) taskInfo->run);
	}
	
	
	inline void createdTask(__attribute__((unused)) Task *task, __attribute__((unused)) task_id_t taskId)
	{
	}
	
	
	inline void exitAddTask(__attribute__((unused)) task_id_t taskId)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_NO_STATE;
		
		Extrae_emit_CombinedEvents ( &ce );
	}
}


#endif // INSTRUMENT_EXTRAE_ADD_TASK_HPP
