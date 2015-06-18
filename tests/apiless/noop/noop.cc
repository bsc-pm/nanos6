#include <cstdlib>

#include "tests/infrastructure/TestAnyProtocolProducer.hpp"


TestAnyProtocolProducer tap;


static void endOfTest()
{
	tap.success();
	tap.end();
}


int main(int argc, char **argv)
{
	tap.registerNewTests(2);
	tap.begin();
	
	int rc = atexit(endOfTest);
	if (rc == 0) {
		tap.success();
	} else {
		tap.failure();
	}
	
	return 0;
}

