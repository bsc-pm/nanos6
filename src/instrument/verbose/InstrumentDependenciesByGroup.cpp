#include <cassert>

#include "InstrumentDependenciesByGroup.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void beginAccessGroup(
		__attribute__((unused)) task_id_t parentTaskId,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) bool sequenceIsEmpty,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}


	void addTaskToAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}


	void removeTaskFromAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

}
