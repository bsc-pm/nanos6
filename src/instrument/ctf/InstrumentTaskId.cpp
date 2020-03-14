#include "InstrumentTaskId.hpp"

namespace Instrument {
	std::atomic<uint32_t> _nextTaskId(1);
}
