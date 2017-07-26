/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentProfile.hpp"

#include "system/RuntimeInfo.hpp"


namespace Instrument {
	void initialize()
	{
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "profile");
		Profile::init();
	}
	
	
	void shutdown()
	{
		Profile::shutdown();
	}
	
	
}
