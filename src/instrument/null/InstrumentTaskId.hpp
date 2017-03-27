#ifndef INSTRUMENT_NULL_TASK_ID_HPP
#define INSTRUMENT_NULL_TASK_ID_HPP


namespace Instrument {
	//! This is the default task identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct task_id_t {
	};
}

#endif // INSTRUMENT_NULL_TASK_ID_HPP
