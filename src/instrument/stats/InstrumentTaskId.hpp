#ifndef INSTRUMENT_STATS_TASK_ID_HPP
#define INSTRUMENT_STATS_TASK_ID_HPP


namespace Instrument {
	namespace Stats {
		struct TaskTypeAndTimes;
	}
	//! This is the default task identifier for the instrumentation.
	typedef Stats::TaskTypeAndTimes *task_id_t;
}

#endif // INSTRUMENT_STATS_TASK_ID_HPP
