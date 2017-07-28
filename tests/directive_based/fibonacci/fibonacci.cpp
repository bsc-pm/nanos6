/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"


#if TEST_LESS_THREADS
#define N 14
#else
#define N 26
#endif

#define INTEGER unsigned long


TestAnyProtocolProducer tap;


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
	
	#pragma oss task shared(result1) label(fibonacci)
	fibonacci(index-1, &result1);
	
	#pragma oss task shared(result2) label(fibonacci)
	fibonacci(index-2, &result2);
	
	#pragma oss taskwait
	*resultPointer = result1 + result2;
}


int main(int argc, char **argv) {
	tap.registerNewTests(1);
	tap.begin();
	
	INTEGER result;
	
	Timer timer;
	
	#pragma oss task shared(result) label(fibonacci)
	fibonacci(N, &result);
	
	#pragma oss taskwait
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	tap.evaluate(result == TemplatedFibonacci<N>::_value, "Check if the result is correct");
	
	tap.end();
	
	return 0;
}
