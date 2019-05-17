/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentProfile.hpp"

#include "system/RuntimeInfo.hpp"


namespace Instrument {
	extern bool _profilingIsReady;
	
	
	void initialize()
	{
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "profile");
		Profile::init();
		_profilingIsReady = true;
	}
	
	
	void shutdown()
	{
		Profile::shutdown();
		_profilingIsReady = false;
	}
	
	
}

