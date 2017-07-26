/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_PATH_LENGTH_HPP
#define INSTRUMENT_GRAPH_PATH_LENGTH_HPP


#include "InstrumentTaskId.hpp"


namespace Instrument {
	namespace Graph {
		void findTopmostTasksAndPathLengths(task_id_t startingTaskId = 0);
	}
}

#endif // INSTRUMENT_GRAPH_PATH_LENGTH_HPP
