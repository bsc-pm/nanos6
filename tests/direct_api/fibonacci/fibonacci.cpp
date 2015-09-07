#include "api/nanos6_rt_interface.h"
#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__ (__FILE__ ":" TOSTRING(__LINE__))


#define N 26
#define INTEGER unsigned long


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


template <unsigned long index>
struct TemplatedFibonacci {
	enum { _value = TemplatedFibonacci<index-1>::_value + TemplatedFibonacci<index-2>::_value };
};

template <>
struct TemplatedFibonacci<0> {
	enum { _value = 0 };
};

template <>
struct TemplatedFibonacci<1> {
	enum { _value = 1 };
};


extern "C" struct fibonacci_args_block_t {
	INTEGER _index;
	INTEGER *_resultPointer;
};


static void fibonacci(INTEGER index, INTEGER *resultPointer);


static void fibonacci_wrapper(void *argsBlock)
{
	fibonacci_args_block_t *fibonacci_args_block = (fibonacci_args_block_t *) argsBlock;
	
	fibonacci(fibonacci_args_block->_index, fibonacci_args_block->_resultPointer);
}

static void fibonacci_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void fibonacci_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info fibonacci_info = {
	fibonacci_wrapper,
	fibonacci_register_depinfo,
	fibonacci_register_copies,
	"fibonacci",
	"fibonacci_source_line"
};


void fibonacci(INTEGER index, INTEGER *resultPointer) {
	if (index <= 1) {
		*resultPointer = index;
		return;
	}
	
	INTEGER result1;
	fibonacci_args_block_t *fibonacci_args_block1;
	void *fibonacciTask1 = nullptr;
	static nanos_task_invocation_info fibonacci_invocation_info1 = {
		__FILE_LINE__
	};
	nanos_create_task(&fibonacci_info, &fibonacci_invocation_info1, sizeof(fibonacci_args_block_t), (void **) &fibonacci_args_block1, &fibonacciTask1);
	fibonacci_args_block1->_index = index-1;
	fibonacci_args_block1->_resultPointer = &result1;
	nanos_submit_task(fibonacciTask1);
	
	INTEGER result2;
	fibonacci_args_block_t *fibonacci_args_block2;
	void *fibonacciTask2 = nullptr;
	static nanos_task_invocation_info fibonacci_invocation_info2 = {
		__FILE_LINE__
	};
	nanos_create_task(&fibonacci_info, &fibonacci_invocation_info2, sizeof(fibonacci_args_block_t), (void **) &fibonacci_args_block2, &fibonacciTask2);
	fibonacci_args_block2->_index = index-2;
	fibonacci_args_block2->_resultPointer = &result2;
	nanos_submit_task(fibonacciTask2);
	
	nanos_taskwait(__FILE_LINE__);
	*resultPointer = result1 + result2;
}


int main(int argc, char **argv) {
	initializationTimer.stop();
	
	tap.registerNewTests(1);
	tap.begin();
	
	INTEGER result;
	
	Timer timer;
	
	fibonacci_args_block_t *fibonacci_args_block;
	void *fibonacciTask = nullptr;
	static nanos_task_invocation_info fibonacci_invocation_info3 = {
		__FILE_LINE__
	};
	nanos_create_task(&fibonacci_info, &fibonacci_invocation_info3, sizeof(fibonacci_args_block_t), (void **) &fibonacci_args_block, &fibonacciTask);
	fibonacci_args_block->_index = N;
	fibonacci_args_block->_resultPointer = &result;
	nanos_submit_task(fibonacciTask);
	
	nanos_taskwait(__FILE_LINE__);
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	tap.evaluate(result == TemplatedFibonacci<N>::_value, "Check if the result is correct");
	
	shutdownTimer.start();
	
	return 0;
}
