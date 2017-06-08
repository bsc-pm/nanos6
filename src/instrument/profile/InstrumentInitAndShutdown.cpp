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
