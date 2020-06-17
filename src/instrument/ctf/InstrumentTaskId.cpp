#include "InstrumentTaskId.hpp"

// Reserved ctf2prv task and task type Ids:
//   0 : Idle
//   1 : Runtime
//   2 : Busy Wait

std::atomic<uint32_t> Instrument::task_id_t::_nextTaskId(3);
