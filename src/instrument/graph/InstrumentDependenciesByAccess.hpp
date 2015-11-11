#ifndef INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_HPP
#define INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_HPP


#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>


#include "../InstrumentDependenciesByAccess.hpp"
#include "dependencies/DataAccess.hpp"


namespace Instrument {
	void registerTaskAccess(task_id_t taskId, DataAccess::type_t accessType, void *start, size_t length);
}


#endif // INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_HPP
