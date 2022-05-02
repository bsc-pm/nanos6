/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

#include "instrument/api/InstrumentDependencySubsystemEntryPoints.hpp"
#include "OVNITrace.hpp"

namespace Instrument {

	inline void enterRegisterTaskDataAcesses()
	{
		OVNI::registerAccessesEnter();
	}

	inline void exitRegisterTaskDataAcesses()
	{
		OVNI::registerAccessesExit();
	}

	inline void enterUnregisterTaskDataAcesses()
	{
		OVNI::unregisterAccessesEnter();
	}

	inline void exitUnregisterTaskDataAcesses()
	{
		OVNI::unregisterAccessesExit();
	}
}

#endif //INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

