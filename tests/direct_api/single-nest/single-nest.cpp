#include <atomic>
#include <cassert>
#include <cstdio>
#include <sstream>

#include <string.h>
#include <unistd.h>

#include "api/nanos6_rt_interface.h"
#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__ (__FILE__ ":" TOSTRING(__LINE__))


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
	std::atomic<bool> _theTaskHasFinished;
	
	OuterExplicitTaskChecker():
		_mainHasFinished(false), _theTaskHasFinished(false)
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


static void outer_explicit_task_checker_wrapper(void *argsBlock)
{
	OuterExplicitTaskChecker **outer_explicit_task_checker = (OuterExplicitTaskChecker **) argsBlock;
	
	(*outer_explicit_task_checker)->body();
}

static void outer_explicit_task_checker_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void outer_explicit_task_checker_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info outer_explicit_task_checker_info = {
	outer_explicit_task_checker_wrapper,
	outer_explicit_task_checker_register_depinfo,
	outer_explicit_task_checker_register_copies,
	"outer_explicit_task_checker",
	"outer_explicit_task_checker_source_line"
};




int main(int argc, char **argv)
{
	initializationTimer.stop();
	
	tap.registerNewTests(2);
	tap.begin();
	
	theOuterTaskHasFinished = false;
	
	OuterExplicitTaskChecker *outer_explicit_task_checker = new OuterExplicitTaskChecker();
	OuterExplicitTaskChecker **outer_explicit_task_checker_param;
	void *outer_explicit_task_checker_task = nullptr;
	static nanos_task_invocation_info outer_explicit_task_checker_invocation_info = {
		__FILE_LINE__
	};
	nanos_create_task(&outer_explicit_task_checker_info, &outer_explicit_task_checker_invocation_info, sizeof(OuterExplicitTaskChecker *), (void **) &outer_explicit_task_checker_param, &outer_explicit_task_checker_task);
	*outer_explicit_task_checker_param = outer_explicit_task_checker;
	nanos_submit_task(outer_explicit_task_checker_task);
	
	outer_explicit_task_checker->_mainHasFinished = true; // Well, almost. It has continued after adding the task
	
	shutdownTimer.start();
	
	return 0;
}

