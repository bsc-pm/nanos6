#include "DeviceHierarchicalScheduler.hpp"

#include "../DefaultScheduler.hpp"
#include "../FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "../FIFOScheduler.hpp"
#include "../ImmediateSuccessorScheduler.hpp"
#include "../ImmediateSuccessorWithPollingScheduler.hpp"
#include "../PriorityScheduler.hpp"
#include "../SchedulerInterface.hpp"

#include "lowlevel/EnvironmentVariable.hpp"

#include <cassert>


DeviceHierarchicalScheduler::DeviceHierarchicalScheduler() : SchedulerInterface()
{
	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");

	if (schedulerName.getValue() == "default") {
		_CPUScheduler = new DefaultScheduler();
	} else if (schedulerName.getValue() == "fifo") {
		_CPUScheduler = new FIFOScheduler();
	} else if (schedulerName.getValue() == "immediatesuccessor") {
		_CPUScheduler = new ImmediateSuccessorScheduler();
	} else if (schedulerName.getValue() == "iswp") {
		_CPUScheduler = new ImmediateSuccessorWithPollingScheduler();
	} else if (schedulerName.getValue() == "fifoiswp") {
		_CPUScheduler = new FIFOImmediateSuccessorWithPollingScheduler();
	} else if (schedulerName.getValue() == "priority") {
		_CPUScheduler = new PriorityScheduler();
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName.getValue() << "', using default instead." << std::endl;
		_CPUScheduler = new DefaultScheduler();
	}
}

DeviceHierarchicalScheduler::~DeviceHierarchicalScheduler()
{
	delete _CPUScheduler;
}


ComputePlace * DeviceHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
{
	return _CPUScheduler->addReadyTask(task, hardwarePlace, hint);
}


void DeviceHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_CPUScheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *DeviceHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask)
{
	return _CPUScheduler->getReadyTask(hardwarePlace, currentTask);
}


ComputePlace *DeviceHierarchicalScheduler::getIdleComputePlace(bool force)
{
	return _CPUScheduler->getIdleComputePlace(force);
}

void DeviceHierarchicalScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	_CPUScheduler->disableComputePlace(hardwarePlace);
}

void DeviceHierarchicalScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	_CPUScheduler->enableComputePlace(hardwarePlace);
}
