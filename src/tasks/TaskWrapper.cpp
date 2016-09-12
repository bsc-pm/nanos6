#include "TaskWrapper.hpp"

#include "api/nanos6_rt_interface.h"

#include <cassert>


static void task_environment_wrapper_run(__attribute__((unused)) void *args)
{
	assert("Attempt to run the body of a task environment wrapper" == nullptr);
}


nanos_task_info TaskWrapper::_wrapperTaskInfo = {
	task_environment_wrapper_run,
	nullptr,
	nullptr,
	"nanos task environment wrapper",
	"",
	nullptr
};

nanos_task_invocation_info TaskWrapper::_wrapperTaskInvocationInfo = {
	""
};

