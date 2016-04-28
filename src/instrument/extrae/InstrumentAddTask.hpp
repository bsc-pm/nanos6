#ifndef INSTRUMENT_EXTRAE_ADD_TASK_HPP
#define INSTRUMENT_EXTRAE_ADD_TASK_HPP


//#include "../InstrumentAddTask.hpp"
#include <InstrumentTaskId.hpp>
#include "InstrumentExtrae.hpp"


class Task;


namespace Instrument {

	inline task_id_t enterAddTask(__attribute__((unused)) nanos_task_info *taskInfo, __attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo)
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

//fprintf(stderr,"XTERUEL: Extrae_emit_CombinedEvents %s:%d:%d\n","enterAddTask", _runtimeState, NANOS_CREATION ); //FIXME:debug
      Extrae_emit_CombinedEvents ( &ce );

		if (taskInfo) {
         const char * label = taskInfo->task_label? taskInfo->task_label: taskInvokationInfo->invocation_source;
         _userFunctionMap[(void *)taskInfo->run] = label;

//fprintf(stderr,"XTERUEL: addr =%p, declaration= %s, label=%s, invocation = %s\n", taskInfo->run, taskInfo->declaration_source, taskInfo->task_label, taskInvokationInfo->invocation_source); // FIXME:debug
//fprintf(stderr,"XTERUEL: addr =%p, label=%s\n", taskInfo->run, label); // FIXME:debug
         
		}

		return task_id_t();
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

//fprintf(stderr,"XTERUEL: Extrae_emit_CombinedEvents %s:%d:%d\n","exitAddTask",_runtimeState, NANOS_NO_STATE); //FIXME:debug
      Extrae_emit_CombinedEvents ( &ce );
	}
}


#endif // INSTRUMENT_EXTRAE_ADD_TASK_HPP
