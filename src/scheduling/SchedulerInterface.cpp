#include "SchedulerInterface.hpp"

#include <cassert>


void SchedulerInterface::disableHardwarePlace(__attribute__((unused)) HardwarePlace *hardwarePlace)
{
}


void SchedulerInterface::enableHardwarePlace(__attribute__((unused)) HardwarePlace *hardwarePlace)
{
}


bool SchedulerInterface::requestPolling(HardwarePlace *hardwarePlace, polling_slot_t *pollingSlot)
{
	assert(pollingSlot != nullptr);
	assert(pollingSlot->_task == nullptr);
	
	// Default implementation: attempt to get a ready task and fail if not possible
	Task *task = getReadyTask(hardwarePlace);
	
	if (task != nullptr) {
		Task *expected = nullptr;
		
		pollingSlot->_task.compare_exchange_strong(expected, task);
		assert(expected == nullptr);
		
		return true;
	} else {
		return false;
	}
}


bool SchedulerInterface::releasePolling(__attribute__((unused)) HardwarePlace *hardwarePlace, __attribute__((unused)) polling_slot_t *pollingSlot)
{
	// The default implementation should never be called if there is a default implementation of requestPolling
	// otherwise there should be an implementation of this method that matches requestPolling
	assert(false);
	return true;
}

