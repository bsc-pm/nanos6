/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

#include "instrument/api/InstrumentDependencySubsystemEntryPoints.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void enterRegisterTaskDataAcesses()
	{
		Ovni::registerAccessesEnter();
	}

	inline void exitRegisterTaskDataAcesses()
	{
		Ovni::registerAccessesExit();
	}

	inline void enterUnregisterTaskDataAcesses()
	{
		Ovni::unregisterAccessesEnter();
	}

	inline void exitUnregisterTaskDataAcesses()
	{
		Ovni::unregisterAccessesExit();
	}
}

#endif //INSTRUMENT_OVNI_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

