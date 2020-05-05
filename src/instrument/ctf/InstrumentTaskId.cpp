#include "InstrumentTaskId.hpp"

namespace Instrument {
	// task Id = 1 is reserved for "Runtime mode" in Paraver Views
	std::atomic<uint32_t> _nextTaskId(2);

	SpinLock globalTaskLabelLock;
	// taskTypeId 0 reserved for extrae "End"
	// taskTypeId 1 reserved for extrae "cpu idle"
	uint32_t _nextTaskTypeId = 2; //protected with globalTaskTypeIdsLock
	taskLabelMap_t globalTaskLabelMap; // protected with globalTaskTypeIdsLock
}
