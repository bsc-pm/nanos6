/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "CPUActivation.hpp"
#include "DefaultThreadManagerPolicy.hpp"

#include "tasks/Task.hpp"


bool DefaultThreadManagerPolicy::checkIfMustRunInline(Task *replacementTask, Task *currentTask, CPU *cpu)
{
	return CPUActivation::acceptsWork(cpu) && (replacementTask->getParent() == currentTask);
}

bool DefaultThreadManagerPolicy::checkIfUnblockedMustPreemtUnblocker(
	__attribute__((unused)) Task *unblockerTask,
	__attribute__((unused)) Task *unblockedTask,
	__attribute__((unused)) CPU *cpu
) {
	return true;
}

