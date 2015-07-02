#include "system/ompss/AddTask.hpp"
#include "system/ompss/TaskWait.hpp"
#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"


#define N 5


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


template <typename INTEGER>
class FibonacciTask: public Task {
	INTEGER _index;
	INTEGER &_resultReference;
	
public:
	FibonacciTask(INTEGER index, INTEGER &resultReference)
		: Task(nullptr), _index(index), _resultReference(resultReference)
	{
	}
	
	virtual void body()
	{
		if (_index <= 1) {
			_resultReference = _index;
			return;
		}
		
		INTEGER value1, value2;
		
		FibonacciTask *fib1 = new FibonacciTask(_index-1, value1);
		ompss::addTask(fib1);
		FibonacciTask *fib2 = new FibonacciTask(_index-2, value2);
		ompss::addTask(fib2);
		
		ompss::taskWait();
		_resultReference = value1 + value2;
	}
};


int main(int argc, char **argv) {
	initializationTimer.stop();
	
	tap.registerNewTests(1);
	tap.begin();
	
	unsigned long result;
	
	Timer timer;
	
	FibonacciTask<unsigned long> *fib = new FibonacciTask<unsigned long>(N, result);
	ompss::addTask(fib);
	ompss::taskWait();
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	tap.evaluate(result == TemplatedFibonacci<N>::_value, "Check if the result is correct");
	
	shutdownTimer.start();
	
	return 0;
}
