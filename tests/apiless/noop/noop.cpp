#include "tests/infrastructure/TestAnyProtocolProducer.hpp"


// This is shared with tests/infrastructure/ProgramLifecycle.cc
extern TestAnyProtocolProducer tap;


// This function is called after the runtime has shut down
void shutdownTests()
{
	tap.success("Could exit the runtime");
	tap.end();
}


int main(int argc, char **argv)
{
	tap.registerNewTests(1); // Only the shutdown test
	tap.begin();
	
	return 0;
}

