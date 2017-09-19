#ifndef TASK_IMPLEMENTATION_HPP
#define TASK_IMPLEMENTATION_HPP


#include "Task.hpp"

#include <TaskDataAccessesImplementation.hpp>

#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>


inline Task::Task(
	void *argsBlock,
	nanos_task_info *taskInfo,
	nanos_task_invocation_info *taskInvokationInfo,
	Task *parent,
	Instrument::task_id_t instrumentationTaskId,
	size_t flags
)
	: _argsBlock(argsBlock),
	_taskInfo(taskInfo),
	_taskInvokationInfo(taskInvokationInfo),
	_countdownToBeWokenUp(1),
	_parent(parent),
	_thread(nullptr),
	_dataAccesses(),
	_flags(flags),
	_predecessorCount(0),
	_instrumentationTaskId(instrumentationTaskId),
	_schedulerInfo(nullptr)
{
	if (parent != nullptr) {
		parent->addChild(this);
	}
}




#endif // TASK_IMPLEMENTATION_HPP
