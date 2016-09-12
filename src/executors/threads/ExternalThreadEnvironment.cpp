#include "ExternalThreadEnvironment.hpp"
#include "tasks/TaskWrapper.hpp"

#include <cassert>

#include <pthread.h>


__thread ExternalThreadEnvironment *ExternalThreadEnvironment::_taskWrapperEvironment = nullptr;


ExternalThreadEnvironment::ExternalThreadEnvironment()
	: EssentialThreadEnvironment(), _cpuDependencyData()
{
	_task = new TaskWrapper();
	_task->setThread(this);
	_pthread = pthread_self();
}


ExternalThreadEnvironment::~ExternalThreadEnvironment()
{
	if (_task != nullptr) {
		TaskWrapper *_taskWrapper = static_cast<TaskWrapper *>(_task);
		if (_taskWrapper != nullptr) {
			delete _taskWrapper;
		} else {
			assert("Attempt to destroy a thread that has a task assigned" && (_task == nullptr));
		}
	}
}
