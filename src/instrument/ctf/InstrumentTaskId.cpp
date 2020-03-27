#include "InstrumentTaskId.hpp"

namespace Instrument {
	std::atomic<uint32_t> _nextTaskId(1);

	SpinLock globalTaskLabelLock;
	uint32_t _nextTaskTypeId = 1; //protected with globalTaskTypeIdsLock
	taskLabelMap_t globalTaskLabelMap; // protected with globalTaskTypeIdsLock
}
