/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_NULL_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

#include "../api/InstrumentDependencySubsystemEntryPoints.hpp"


namespace Instrument {

	inline void enterRegisterTaskDataAcesses() {}

	inline void exitRegisterTaskDataAcesses() {}

	inline void enterUnregisterTaskDataAcesses() {}

	inline void exitUnregisterTaskDataAcesses() {}

}

#endif //INSTRUMENT_NULL_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

