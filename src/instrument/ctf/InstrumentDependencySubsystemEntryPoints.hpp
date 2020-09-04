/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_CTF_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

#include "CTFTracepoints.hpp"
#include "instrument/api/InstrumentDependencySubsystemEntryPoints.hpp"


namespace Instrument {

	inline void enterRegisterTaskDataAcesses()
	{
		tp_dependency_register_enter();
	}

	inline void exitRegisterTaskDataAcesses()
	{
		tp_dependency_register_exit();
	}

	inline void enterUnregisterTaskDataAcesses()
	{
		tp_dependency_unregister_enter();
	}

	inline void exitUnregisterTaskDataAcesses()
	{
		tp_dependency_unregister_exit();
	}

}

#endif //INSTRUMENT_CTF_DEPENDENCY_SUBSYTEM_ENTRY_POINTS_HPP

