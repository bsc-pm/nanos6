#ifndef TEST_ANY_PROTOCOL_PRODUCER_HPP
#define TEST_ANY_PROTOCOL_PRODUCER_HPP

#include <cxxabi.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

#include <sys/time.h>


//! \brief a class for generating the TAP testing protocol that the autotools recognizes
class TestAnyProtocolProducer {
private:
	int _testCount;
	int _currentTest;
	bool _hasFailed;
	std::string _component;
	
	void emitOutcome(std::string const &outcome, std::string const &detail, std::string const &special = "")
	{
		std::cout << outcome << " " << _currentTest;
		if (_component != "") {
			std::cout << " " << _component << ":";
		}
		if (detail != "") {
			std::cout << " " << detail;
		}
		if (special != "") {
			std::cout << " # " << special;
		}
		std::cout << std::endl;
		_currentTest++;
	}
	
public:
	TestAnyProtocolProducer()
		: _testCount(0), _currentTest(0), _hasFailed(false)
	{
	}
	
	//! \brief register new tests
	//! If possible, register the total number of tests before starting
	//!
	//! \param in count the number of new tests to register
	void registerNewTests(int count=1)
	{
		_testCount += count;
	}
	
	//! \brief set up the name of the component that is being tested
	//!
	//! \param in component name of the component, for instance the result of typeid(_a_class_or_object_).name()
	void setComponent(std::string const &component)
	{
		int status;
		char *demangledComponent = abi::__cxa_demangle(component.c_str(), 0, 0, &status);
		
		if (status == 0) {
			_component = std::string(demangledComponent);
			free(demangledComponent);
		} else {
			_component = component;
		}
	}
	
	//! \brief start the set of tests (sequentially)
	//! Each test is run one after the other and its result is indicated by just calling succecc, failure, todo, skip or evaluate
	void begin()
	{
		if (_testCount != 0) {
			std::cout << "1.." << _testCount << std::endl;
		}
		_currentTest = 1;
	}
	
	//! \brief finish the set of tests (sequentially)
	void end()
	{
	}
	
	
	//! \brief indicate that the current test in sequential order was successful
	//!
	//! \param detail in optionally any additional information about the outcome of the test
	void success(std::string const &detail="")
	{
		emitOutcome("ok", detail);
	}
	
	//! \brief indicate that the current test in sequential order failed
	//!
	//! \param detail in optionally any additional information about the outcome of the test
	void failure(std::string const &detail="")
	{
		emitOutcome("not ok", detail);
		_hasFailed = true;
	}
	
	//! \brief indicate that the current test in sequential order is yet to be implemented
	//!
	//! \param detail in optionally any additional information about the outcome of the test
	void todo(std::string const &detail="")
	{
		emitOutcome("not ok", detail, "TODO");
	}
	
	//! \brief indicate that the current test in sequential order has been skipped
	//!
	//! \param detail in optionally any additional information about why it was skipped
	void skip(std::string const &detail="")
	{
		emitOutcome("ok", detail, "SKIP");
	}
	
	//! \brief indicate that the set of tests will stop here (even if not all of them have been run)
	//!
	//! \param detail in optionally any additional information about the reason
	void bailOut(std::string const &detail="")
	{
		std::cout << "Bail out!";
		if (detail != "") {
			std::cout << " " << detail;
		}
		std::cout << std::endl;
	}
	
	
	//! \brief get the outcome of the current test through a condition
	//!
	//! \param condition in true if the test was successful, false otherwise
	//! \param detail in optionally any additional information about the test
	void evaluate(bool condition, std::string const &detail="")
	{
		if (condition) {
			success(detail);
		} else {
			failure(detail);
		}
	}
	
	//! \brief check that a condition is satisfied in a given amount of time
	//!
	//! \param condition in true if the test was successful, false otherwise
	//! \param microseconds grace period to assert the condition
	//! \param detail in optionally any additional information about the test
	void timedEvaluate(std::function<bool ()> condition, long microseconds, std::string const &detail="")
	{
		struct timeval start, end, maximum;
		
		int rc = gettimeofday(&start, nullptr);
		if (rc != 0) {
			failure("Failed to get time for a timed check");
			return;
		}
		
		maximum.tv_sec = start.tv_sec;
		maximum.tv_usec = start.tv_usec + microseconds;
		if (maximum.tv_usec >= 1000000) {
			maximum.tv_sec += 1;
			maximum.tv_usec -= 1000000;
		}
		
		while (!condition()) {
			rc = gettimeofday(&end, nullptr);
			bool timeout = (end.tv_sec > maximum.tv_sec);
			timeout = timeout || ((end.tv_sec == maximum.tv_sec) && (end.tv_usec > maximum.tv_usec));
			
			if (timeout) {
				if (condition()) {
					success(detail);
					return;
				} else {
					failure(detail);
					return;
				}
			}
		}
		
		success(detail);
	}
	
	//! \brief exit abruptly if any of the tests up to that point has failed
	//! Possibly because it can cause the program to fail, or the rest of the tests will fail
	void bailOutAndExitIfAnyFailed()
	{
		if (_hasFailed) {
			bailOut("to avoid further errors");
			std::exit(1);
		}
	}
	
	//! \brief check if any of the previous tests has been reported as a failure
	bool hasFailed() const
	{
		return _hasFailed;
	}
	
	//! \brief ignore any previous failure, but still report them
	void clearFailureMark()
	{
		_hasFailed = false;
	}
	
	//! \brief emit an additional comment in the output
	template <typename T1>
	void emitDiagnostic(T1 const &diagnostic)
	{
		std::cout << "# " << diagnostic << std::endl;
	}
	
	template <typename T1, typename T2>
	void emitDiagnostic(T1 const &diagnostic1, T2 const &diagnostic2)
	{
		std::cout << "# " << diagnostic1 << diagnostic2 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3>
	void emitDiagnostic(T1 const &diagnostic1, T2 const &diagnostic2, T3 const &diagnostic3)
	{
		std::cout << "# " << diagnostic1 << diagnostic2 << diagnostic3 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4>
	void emitDiagnostic(T1 const &diagnostic1, T2 const &diagnostic2, T3 const &diagnostic3, T4 const &diagnostic4)
	{
		std::cout << "# " << diagnostic1 << diagnostic2 << diagnostic3 << diagnostic4 << std::endl;
	}
	
};



#endif // TEST_ANY_PROTOCOL_PRODUCER_HPP
