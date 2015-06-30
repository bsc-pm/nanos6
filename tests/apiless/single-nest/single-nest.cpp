#include <atomic>
#include <cassert>
#include <cstdio>
#include <sstream>

#include <string.h>
#include <unistd.h>

#include "system/ompss/AddTask.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"


extern TestAnyProtocolProducer tap;
std::atomic<bool> theOuterTaskHasFinished;


void shutdownTests()
{
	/* Test2 */ tap.evaluate(theOuterTaskHasFinished, "Evaluating whether the runtime exits after the regular task");
	
	assert(theOuterTaskHasFinished);
	tap.end();
}


class OuterExplicitTask: public Task {
public:
	std::atomic<bool> _mainHasFinished;
	std::atomic<bool> _theTaskHasFinished;
	
	OuterExplicitTask():
		Task(nullptr), _mainHasFinished(false), _theTaskHasFinished(false)
	{
	}
	
	virtual void body()
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
	tap.registerNewTests(2);
	tap.begin();
	
	theOuterTaskHasFinished = false;
	
	OuterExplicitTask *theOuterTask = new OuterExplicitTask();
	ompss::addTask(theOuterTask);
	
	theOuterTask->_mainHasFinished = true; // Well, almost. It has continued after adding the task
	
	return 0;
}

