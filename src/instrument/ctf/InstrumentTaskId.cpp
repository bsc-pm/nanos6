#include "InstrumentTaskId.hpp"

namespace Instrument {
	// Reserved ctf2prv task and task type Ids:
	//   0 : Idle
	//   1 : Runtime
	//   2 : Busy Wait

	std::atomic<uint32_t> _nextTaskId(3);

	SpinLock globalTaskLabelLock;
	uint32_t _nextTaskTypeId = 3; //protected with globalTaskTypeIdsLock
	taskLabelMap_t globalTaskLabelMap; // protected with globalTaskTypeIdsLock
}
