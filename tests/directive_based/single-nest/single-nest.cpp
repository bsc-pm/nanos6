#include <atomic>
#include <cassert>
#include <cstdio>
#include <sstream>

#include <string.h>
#include <unistd.h>

#include "infrastructure/ProgramLifecycle.hpp"
#include "infrastructure/TestAnyProtocolProducer.hpp"
#include "infrastructure/Timer.hpp"


extern TestAnyProtocolProducer tap;
std::atomic<bool> theOuterTaskHasFinished;


void shutdownTests()
{
	/* Test2 */ tap.evaluate(theOuterTaskHasFinished, "Evaluating whether the runtime exits after the regular task");
	
	assert(theOuterTaskHasFinished);
	tap.end();
}


class OuterExplicitTaskChecker {
public:
	std::atomic<bool> _mainHasFinished;
	
	OuterExplicitTaskChecker():
		_mainHasFinished(false)
	{
	}
	
	void body()
	{
		long waitIncrement = 1;
		for (long i=0; i < 8192; i+= waitIncrement) {
			if (!_mainHasFinished) {
				std::ostringstream oss;
				oss << "Still waiting for main to finish after " << i << " ms";
				/* Test1 */ tap.emitDiagnostic(oss.str());
			} else {
				break;
			}
			struct timespec ts = { 0, waitIncrement*1000 };
			int rc = nanosleep(&ts, nullptr);
			
			if (rc != 0) {
				/* Test1 */ tap.failure(std::string(strerror(errno)));
				/* Test1 */ tap.bailOut();
				return;
			}
		}
		
		/* Test1 */ tap.evaluate(_mainHasFinished, "Evaluating within a regular task whether the main task has finished in a reasonable amount of time");
		
		theOuterTaskHasFinished = true;
	}
};



int main(int argc, char **argv)
{
	initializationTimer.stop();
	
	tap.registerNewTests(2);
	tap.begin();
	
	theOuterTaskHasFinished = false;
	
	OuterExplicitTaskChecker *outer_explicit_task_checker = new OuterExplicitTaskChecker();
	
	#pragma oss task
	outer_explicit_task_checker->body();
	
	outer_explicit_task_checker->_mainHasFinished = true; // Well, almost. It has continued after adding the task
	
	shutdownTimer.start();
	
	return 0;
}

