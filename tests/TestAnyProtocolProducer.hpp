/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TEST_ANY_PROTOCOL_PRODUCER_HPP
#define TEST_ANY_PROTOCOL_PRODUCER_HPP

#include <cxxabi.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <sstream>

#include <sys/time.h>

#if __cplusplus >= 201103L
#include <mutex>
#else

#include <pthread.h>

namespace std {
	struct mutex {
		pthread_mutex_t _mutex;
		
		mutex()
		{
			pthread_mutex_init(&_mutex, 0);
		}
		
		void lock()
		{
			pthread_mutex_lock(&_mutex);
		}
		void unlock()
		{
			pthread_mutex_unlock(&_mutex);
		}
	};
	
	template <typename T>
	struct lock_guard {
		mutex &_mutex;
		
		lock_guard(mutex &mutex)
			: _mutex(mutex)
		{
			_mutex.lock();
		}
		
		~lock_guard()
		{
			_mutex.unlock();
		}
	};
}

#endif


//! \brief a class for generating the TAP testing protocol that the autotools recognizes
class TestAnyProtocolProducer {
private:
	int _testCount;
	int _currentTest;
	bool _hasFailed;
	std::string _component;
	
	std::mutex _outputAndCounterMutex;
	
	void emitOutcome(std::string const &outcome, std::string const &detail, std::string const &special = "")
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
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
	}
	
public:
	TestAnyProtocolProducer()
		: _testCount(0), _currentTest(0), _hasFailed(false), _outputAndCounterMutex()
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
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		{
			_currentTest = 1;
		}
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
	
	//! \brief indicate that the current test in sequential order failed but that it was an expected failure
	//!
	//! \param[in] detail information about the outcome of the test
	//! \param[in] weakDetail information about why the test is weak 
	void weakFailure(std::string const &detail, std::string const &weakDetail)
	{
		std::ostringstream special;
		
		special << "TODO " << weakDetail;
		
		emitOutcome("not ok", detail, special.str());
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
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		{
			std::cout << "Bail out!";
			if (detail != "") {
				std::cout << " " << detail;
			}
			std::cout << std::endl;
		}
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
	
	//! \brief get the outcome of the current test through a condition
	//!
	//! \param[in] condition true if the test was successful, false otherwise
	//! \param[in] detail additional information about the test
	//! \param[in] weakDetail information about why the test is weak 
	void evaluateWeak(bool condition, std::string const &detail, std::string const &weakDetail)
	{
		if (condition) {
			success(detail);
		} else {
			weakFailure(detail, weakDetail);
		}
	}
	
	//! \brief check that a condition is satisfied in a given amount of time
	//!
	//! \param[in] condition true if the test was successful, false otherwise
	//! \param[in] microseconds grace period to assert the condition
	//! \param[in] detail optionally any additional information about the test
	//! \param[in] weak true if a timeout does not necessarily mean incorrectness
	template <typename ConditionType>
	void timedEvaluate(ConditionType condition, long microseconds, std::string const &detail="", bool weak=false)
	{
		struct timeval start, end, maximum;
		
		int rc = gettimeofday(&start, 0);
		if (rc != 0) {
			failure("Failed to get time for a timed check");
			return;
		}
		
		maximum.tv_sec = start.tv_sec;
		maximum.tv_usec = start.tv_usec + microseconds;
		while (maximum.tv_usec >= 1000000L) {
			maximum.tv_sec += 1;
			maximum.tv_usec -= 1000000L;
		}
		
		while (!condition()) {
			rc = gettimeofday(&end, 0);
			bool timeout = (end.tv_sec > maximum.tv_sec);
			timeout = timeout || ((end.tv_sec == maximum.tv_sec) && (end.tv_usec > maximum.tv_usec));
			
			if (timeout) {
				if (condition()) {
					success(detail);
					return;
				} else {
					if (!weak) {
						failure(detail);
					} else {
						weakFailure(detail, "timed out waiting for the condition to be asserted");
					}
					return;
				}
			}
		}
		
		success(detail);
	}
	
	//! \brief check that a condition is kept during a given amount of time
	//!
	//! \param[in] condition true during the time period for the test to be successful
	//! \param[in] microseconds period of time furing which the condition is expected to be asserted
	//! \param[in] detail optionally any additional information about the test
	template <typename ConditionType>
	void sustainedEvaluate(ConditionType condition, long microseconds, std::string const &detail="")
	{
		struct timeval start, end, maximum;
		
		int rc = gettimeofday(&start, 0);
		if (rc != 0) {
			failure("Failed to get time for a timed check");
			return;
		}
		
		maximum.tv_sec = start.tv_sec;
		maximum.tv_usec = start.tv_usec + microseconds;
		while (maximum.tv_usec >= 1000000L) {
			maximum.tv_sec += 1;
			maximum.tv_usec -= 1000000L;
		}
		
		while (condition()) {
			rc = gettimeofday(&end, 0);
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
		
		failure(detail);
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
	
	template <typename T1>
	void emitDiagnostic(T1 v1)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << std::endl;
	}
	
	template <typename T1, typename T2>
	void emitDiagnostic(T1 v1, T2 v2)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << v6 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << v6 << v7 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << v9 << std::endl;
	}
	
	template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
	void emitDiagnostic(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10)
	{
		std::lock_guard<std::mutex> guard(_outputAndCounterMutex);
		std::cout << "# " << v1 << v2 << v3 << v4 << v5 << v6 << v7 << v8 << v9 << v10 << std::endl;
	}
	
};



#endif // TEST_ANY_PROTOCOL_PRODUCER_HPP
