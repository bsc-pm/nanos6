#ifndef TASK_WRAPPER_HPP
#define TASK_WRAPPER_HPP


#include "Task.hpp"


//! \brief a ficticious task to provide an environment for threads that create tasks outside of a task
class TaskWrapper : public Task {
	static nanos_task_info _wrapperTaskInfo;
	static nanos_task_invocation_info _wrapperTaskInvocationInfo;
	
public:
	TaskWrapper()
		: Task(nullptr, &_wrapperTaskInfo, &_wrapperTaskInvocationInfo, nullptr, Instrument::task_id_t())
	{
	}
};



#endif // TASK_WRAPPER_HPP

