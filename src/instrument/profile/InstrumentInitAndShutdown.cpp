#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentProfile.hpp"


namespace Instrument {
	void initialize()
	{
		Profile::init();
	}
	
	
	void shutdown()
	{
		Profile::shutdown();
	}
	
	
}
