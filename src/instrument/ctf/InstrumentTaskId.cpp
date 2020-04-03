#include "InstrumentTaskId.hpp"

namespace Instrument {
	std::atomic<uint32_t> _nextTaskId(1);

	SpinLock globalTaskLabelLock;
	// taskTypeId 0 reserved for extrae "End"
	// taskTypeId 1 reserved for extrae "cpu idle"
	uint32_t _nextTaskTypeId = 2; //protected with globalTaskTypeIdsLock
	taskLabelMap_t globalTaskLabelMap; // protected with globalTaskTypeIdsLock
}
