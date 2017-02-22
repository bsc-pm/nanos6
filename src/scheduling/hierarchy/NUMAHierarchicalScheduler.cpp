#include "HostHierarchicalScheduler.hpp"
#include "NUMAHierarchicalScheduler.hpp"

#include "../DefaultScheduler.hpp"
#include "../FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "../FIFOScheduler.hpp"
#include "../ImmediateSuccessorScheduler.hpp"
#include "../ImmediateSuccessorWithPollingScheduler.hpp"
#include "../PriorityScheduler.hpp"
#include "../Scheduler.hpp"
#include "../SchedulerInterface.hpp"

#include "hardware/HardwareInfo.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

#include <cassert>


NUMAHierarchicalScheduler::NUMAHierarchicalScheduler()
	: _NUMANodeScheduler(HardwareInfo::getMemoryNodeCount())
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();

	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");

	if (schedulerName.getValue() == "default") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new DefaultScheduler();
		}
	} else if (schedulerName.getValue() == "fifo") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new FIFOScheduler();
		}
	} else if (schedulerName.getValue() == "immediatesuccessor") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new ImmediateSuccessorScheduler();
		}
	} else if (schedulerName.getValue() == "iswp") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new ImmediateSuccessorWithPollingScheduler();
		}
	} else if (schedulerName.getValue() == "fifoiswp") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new FIFOImmediateSuccessorWithPollingScheduler();
		}
	} else if (schedulerName.getValue() == "priority") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new PriorityScheduler();
		}
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName.getValue() << "', using default instead." << std::endl;
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new DefaultScheduler();
		}
	}
	
}

NUMAHierarchicalScheduler::~NUMAHierarchicalScheduler()
{
	for (SchedulerInterface *sched : _NUMANodeScheduler) {
		delete sched;
	}
}


ComputePlace * NUMAHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
{
	return _NUMANodeScheduler[0]->addReadyTask(task, hardwarePlace, hint);
}


void NUMAHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_NUMANodeScheduler[0]->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *NUMAHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask)
{
	/* TODO: be a bit more clever */
	return _NUMANodeScheduler[((CPU *)hardwarePlace)->_NUMANodeId]->getReadyTask(hardwarePlace, currentTask);
}


ComputePlace *NUMAHierarchicalScheduler::getIdleComputePlace(bool force)
{
	/* TODO: be a bit more clever */
	ComputePlace *computePlace = nullptr;

	for (SchedulerInterface *sched : _NUMANodeScheduler) {
		computePlace = sched->getIdleComputePlace(force);
		if (computePlace != nullptr) {
			break;
		}
	}

	return computePlace;
}

