#include "InstrumentTaskId.hpp"

namespace Instrument {
	std::atomic<size_t> _nextTaskId(1);
}
