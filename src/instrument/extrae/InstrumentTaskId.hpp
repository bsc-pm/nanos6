#ifndef INSTRUMENT_TASK_ID_HPP
#define INSTRUMENT_TASK_ID_HPP


#include <nanos6.h>

#include "InstrumentExtrae.hpp"


namespace Instrument {
	struct task_id_t {
		nanos_task_info *_taskInfo;
		size_t _taskId;
		int _nestingLevel;
		
		task_id_t()
			: _taskInfo(nullptr), _taskId(~0UL), _nestingLevel(-1)
		{
		}
		
		task_id_t(nanos_task_info *taskInfo, int nestingLevel)
			: _taskInfo(taskInfo), _nestingLevel(nestingLevel)
		{
			_taskId = _nextTaskId++;
		}
		
		
		bool operator==(task_id_t const &other) const
		{
			return (_taskInfo == other._taskInfo);
		}
		
	};
}

#endif // INSTRUMENT_TASK_ID_HPP
