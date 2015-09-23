#include "infrastructure/ProgramLifecycle.hpp"
#include "infrastructure/TestAnyProtocolProducer.hpp"
#include "infrastructure/Timer.hpp"


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


void fibonacci(INTEGER index, INTEGER *resultPointer);


void fibonacci(INTEGER index, INTEGER *resultPointer) {
	if (index <= 1) {
		*resultPointer = index;
		return;
	}
	
	INTEGER result1, result2;
	
	#pragma oss task shared(result1)
	fibonacci(index-1, &result1);
	
	#pragma oss task shared(result2)
	fibonacci(index-2, &result2);
	
	#pragma oss taskwait
	*resultPointer = result1 + result2;
}


int main(int argc, char **argv) {
	initializationTimer.stop();
	
	tap.registerNewTests(1);
	tap.begin();
	
	INTEGER result;
	
	Timer timer;
	
	#pragma oss task shared(result)
	fibonacci(N, &result);
	
	#pragma oss taskwait
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	tap.evaluate(result == TemplatedFibonacci<N>::_value, "Check if the result is correct");
	
	shutdownTimer.start();
	
	return 0;
}
