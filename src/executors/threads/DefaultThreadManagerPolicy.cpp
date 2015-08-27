#include <cassert>

#include "CPUActivation.hpp"
#include "DefaultThreadManagerPolicy.hpp"

#include "tasks/Task.hpp"


bool DefaultThreadManagerPolicy::checkIfMustRunInline (Task *replacementTask, Task *currentTask, CPU *cpu)
{
	return CPUActivation::acceptsWork(cpu) && (replacementTask->getParent() == currentTask);
}

