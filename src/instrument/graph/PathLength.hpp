#ifndef INSTRUMENT_GRAPH_PATH_LENGTH_HPP
#define INSTRUMENT_GRAPH_PATH_LENGTH_HPP


#include "InstrumentTaskId.hpp"


namespace Instrument {
	namespace Graph {
		void findTopmostTasksAndPathLengths(task_id_t startingTaskId = 0);
	}
}

#endif // INSTRUMENT_GRAPH_PATH_LENGTH_HPP
