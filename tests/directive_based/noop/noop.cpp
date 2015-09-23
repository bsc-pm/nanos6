#include "infrastructure/ProgramLifecycle.hpp"
#include "infrastructure/TestAnyProtocolProducer.hpp"
#include "infrastructure/Timer.hpp"


// This is shared with infrastructure/ProgramLifecycle.cc
extern TestAnyProtocolProducer tap;


// This function is called after the runtime has shut down
void shutdownTests()
{
	tap.success("Could exit the runtime");
	tap.end();
}


int main(int argc, char **argv)
{
	initializationTimer.stop();
	
	tap.registerNewTests(1); // Only the shutdown test
	tap.begin();
	
	shutdownTimer.start();
	
	return 0;
}

