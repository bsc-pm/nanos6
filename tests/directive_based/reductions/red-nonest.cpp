/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

#include <math.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"

#include <nanos6/debug.h>


using namespace Functors;


#define SUSTAIN_MICROSECONDS 200000L

//#define FINE_SELF_CHECK


TestAnyProtocolProducer tap;

static int numTests = 0;

static unsigned int ncpus = 0;
static double delayMultiplier = 1.0;


void shutdownTests()
{
}


struct TaskVerifier {
	typedef enum {
		READ,
		WRITE,
		REDUCTION,
		REDUCTION_OTHER
	} type_t;
	
	typedef enum {
		NOT_STARTED,
		STARTED,
		FINISHED
	} status_t;
	
	int _id;
	std::set<int> _runsAfter;
	std::set<int> _runsBefore;
	std::set<int> _runsConcurrentlyWith;
	std::set<int> _runsConcurrentlyWithReduction;
	status_t _status;
	type_t _type;
	int *_variable;
	Atomic<int> *_numConcurrentTasks;
	Atomic<int> *_numConcurrentReductionTasks;
	
private:
	TaskVerifier();
	
public:
	TaskVerifier(int &id, type_t type, int *variable, Atomic<int> *numConcurrentTasks = 0, Atomic<int> *numConcurrentReductionTasks = 0)
		: _id(id++), _runsAfter(), _runsBefore(), _runsConcurrentlyWith(), _runsConcurrentlyWithReduction(),
		_status(NOT_STARTED), _type(type), _variable(variable),
		_numConcurrentTasks(numConcurrentTasks), _numConcurrentReductionTasks(numConcurrentReductionTasks)
	{
	}
	
	
	char const *type2String() const
	{
		switch (_type) {
			case READ:
				return "READ";
			case WRITE:
				return "WRITE";
			case REDUCTION:
				return "REDUCTION";
			case REDUCTION_OTHER:
				return "REDUCTION OTHER";
		}
		
		return "UNKNOWN";
	}
	
	void submit(const std::vector<TaskVerifier *> &verifiers);
	
	void verify(const std::vector<TaskVerifier *> &verifiers)
	{
		assert(_status == NOT_STARTED);
		tap.emitDiagnostic("Task ", _id, " (", type2String(), ") starts");
		_status = STARTED;
		
		for (std::set<int>::const_iterator it = _runsAfter.begin(); it != _runsAfter.end(); it++) {
			int predecessor = *it;
			
			TaskVerifier *predecessorTask = verifiers[predecessor];
			assert(predecessorTask != 0);
			assert(this != predecessorTask);
			{
				std::ostringstream oss;
				oss << "Task " << _id << " must run after task " << predecessorTask->_id;
				tap.evaluate(predecessorTask->_status == FINISHED, oss.str());
			}
		}
		
		if (!_runsConcurrentlyWithReduction.empty()) {
			int nwait = _runsConcurrentlyWithReduction.size() + 1;
			
			assert(_numConcurrentReductionTasks != 0);
			int var = ++(*_numConcurrentReductionTasks);
			
			std::ostringstream oss;
			oss << "Task " << _id << " can run concurrently with all other reduction tasks";
			
			tap.timedEvaluate(
				GreaterOrEqual<Atomic<int>, int>(*_numConcurrentReductionTasks, nwait),
				SUSTAIN_MICROSECONDS * delayMultiplier,
				oss.str()
			);
		}
		
		if (!_runsConcurrentlyWith.empty()) {
			int nwait = _runsConcurrentlyWith.size() + 1;
			
			assert(_numConcurrentTasks != 0);
			int var = ++(*_numConcurrentTasks);
			
			std::ostringstream oss;
			oss << "Task " << _id << " can run concurrently with all other compatible tasks";
			
			tap.timedEvaluate(
				GreaterOrEqual<Atomic<int>, int>(*_numConcurrentTasks, nwait),
				SUSTAIN_MICROSECONDS * delayMultiplier,
				oss.str()
			);
		}
		
		struct timespec delay = { 0, 1000000};
		nanosleep(&delay, &delay);
		
		for (std::set<int>::const_iterator it = _runsBefore.begin(); it != _runsBefore.end(); it++) {
			int successor = *it;
			
			TaskVerifier *successorTask = verifiers[successor];
			assert(successorTask != 0);
			assert(this != successorTask);
			{
				std::ostringstream oss;
				oss << "Task " << _id << " must run before task " << successorTask->_id;
				tap.evaluate(successorTask->_status == NOT_STARTED, oss.str());
			}
		}
		
		_status = FINISHED;
		tap.emitDiagnostic("Task ", _id, " (", type2String(), ") finishes");
	}
};


#pragma oss task in(*variable) label(R)
void verifyRead(int *variable, TaskVerifier *verifier, const std::vector<TaskVerifier *> &verifiers)
{
	assert(verifier != 0);
	verifier->verify(verifiers);
}


#pragma oss task out(*variable) label(W)
void verifyWrite(int *variable, TaskVerifier *verifier, const std::vector<TaskVerifier *> &verifiers)
{
	assert(verifier != 0);
	verifier->verify(verifiers);
}


void verifyReduction(int *variable1, TaskVerifier *verifier, const std::vector<TaskVerifier *> &verifiers)
{
	assert(verifier != 0);
	verifier->verify(verifiers);
}


void TaskVerifier::submit(const std::vector<TaskVerifier *> &verifiers)
{
	switch (_type) {
		case READ:
			verifyRead(_variable, this, verifiers);
			break;
		case WRITE:
			verifyWrite(_variable, this, verifiers);
			break;
		case REDUCTION: {
			int& red_variable = *_variable;
			#pragma oss task reduction(+: red_variable) label(RED)
			verifyReduction(_variable, this, verifiers);
			break;
		}
		case REDUCTION_OTHER: {
			int& red_variable = *_variable;
			#pragma oss task reduction(*: red_variable) label(RED_OTHER)
			verifyReduction(_variable, this, verifiers);
			break;
		}
	}
}


struct VerifierConstraintCalculator {
	typedef enum {
		READERS,
		WRITER,
		REDUCTION
	} access_type_t;
	
	access_type_t _lastAccessType;
	
	const std::vector<TaskVerifier *> *_verifiers;
	
	std::set<int> _lastWriters;
	std::set<int> _lastReaders;
	std::set<int> _newWriters;
	std::set<int> _reducers;
	
	VerifierConstraintCalculator(const std::vector<TaskVerifier *> *verifiers)
		: _lastAccessType(READERS), _verifiers(verifiers), _lastWriters(), _lastReaders(), _newWriters()
	{}
	
	// Fills out the _runsBefore and _runsConcurrentlyWith members of the verifier that is about to exit the current view of the status
	void flush()
	{
		if (_lastAccessType == READERS) {
			// There can only be writers before last access, unless it's the first access
			for (std::set<int>::const_iterator it = _lastWriters.begin(); it != _lastWriters.end(); it++) {
				int writer = *it;
				
				TaskVerifier *writerVerifier = (*_verifiers)[writer];
				assert(writerVerifier != 0);
				
				writerVerifier->_runsBefore = _lastReaders;
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += _lastReaders.size();
#ifdef FINE_SELF_CHECK
				numTests += _lastReaders.size();
#endif
				
				for (std::set<int>::const_iterator it2 = _lastWriters.begin(); it2 != _lastWriters.end(); it2++) {
					int other = *it2;
					
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
				numTests += writerVerifier->_runsConcurrentlyWith.size();
#endif
				numTests +=  writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastWriters.clear();
		} else {
			assert(_lastAccessType == WRITER || _lastAccessType == REDUCTION);
			
			// Readers before last access
			for (std::set<int>::const_iterator it = _lastReaders.begin(); it != _lastReaders.end(); it++) {
				int reader = *it;
				
				TaskVerifier *readerVerifier = (*_verifiers)[reader];
				assert(readerVerifier != 0);
				
				if (_lastAccessType != REDUCTION) {
					readerVerifier->_runsBefore = _newWriters;
					// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
					numTests += _newWriters.size();
#endif
					numTests += _newWriters.size();
				}
				else {
					// Reductions can run concurrently with previous readers
					for (std::set<int>::const_iterator it_red = _newWriters.begin(); it_red != _newWriters.end(); it_red++) {
						readerVerifier->_runsConcurrentlyWith.insert(*it_red);
						(*_verifiers)[*it_red]->_runsConcurrentlyWith.insert(reader);
					}
				}
				
				for (std::set<int>::const_iterator it2 = _lastReaders.begin(); it2 != _lastReaders.end(); it2++) {
					int other = *it2;
					
					if (other != reader)
						readerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
				numTests += readerVerifier->_runsConcurrentlyWith.size();
#endif
				numTests += readerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastReaders.clear();
			
			// Writer(s) before last access (either this or previous set will
			// be non-empty, but not both unless it's the first access)
			for (std::set<int>::const_iterator it = _lastWriters.begin(); it != _lastWriters.end(); it++) {
				int writer = *it;
				
				TaskVerifier *writerVerifier = (*_verifiers)[writer];
				assert(writerVerifier != 0);
				
				if (_lastAccessType != REDUCTION) {
					writerVerifier->_runsBefore = _newWriters;
					// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
					numTests += _newWriters.size();
#endif
					numTests += _newWriters.size();
				}
				else {
					// Reductions can run concurrently with previous writers
					for (std::set<int>::const_iterator it_red = _newWriters.begin(); it_red != _newWriters.end(); it_red++) {
						writerVerifier->_runsConcurrentlyWith.insert(*it_red);
						(*_verifiers)[*it_red]->_runsConcurrentlyWith.insert(writer);
					}
				}
				
				for (std::set<int>::const_iterator it2 = _lastWriters.begin(); it2 != _lastWriters.end(); it2++) {
					int other = *it2;
					
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
				numTests += writerVerifier->_runsConcurrentlyWith.size();
#endif
				numTests += writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastWriters = _newWriters;
			_newWriters.clear();
		}
	}
	
	// Fills out the _runsConcurrentlyWith member of the very last group of accesses
	// and _runsConcurrentlyWithReduction for all accesses
	void flushConcurrent()
	{
		if (_lastAccessType == READERS) {
			for (std::set<int>::const_iterator it = _lastReaders.begin(); it != _lastReaders.end(); it++) {
				int reader = *it;
				
				TaskVerifier *readerVerifier = (*_verifiers)[reader];
				assert(readerVerifier != 0);
				
				for (std::set<int>::const_iterator it2 = _lastReaders.begin(); it2 != _lastReaders.end(); it2++) {
					int other = *it2;
					
					if (other != reader)
						readerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
				numTests += readerVerifier->_runsConcurrentlyWith.size();
#endif
				numTests += readerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
		} else {
			assert(_lastAccessType == WRITER || _lastAccessType == REDUCTION);
			
			for (std::set<int>::const_iterator it = _lastWriters.begin(); it != _lastWriters.end(); it++) {
				int writer = *it;
				TaskVerifier *writerVerifier = (*_verifiers)[writer];
				assert(writerVerifier != 0);
				
				for (std::set<int>::const_iterator it2 = _lastWriters.begin(); it2 != _lastWriters.end(); it2++) {
					int other = *it2;
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
				numTests += writerVerifier->_runsConcurrentlyWith.size();
#endif
				numTests +=  writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
		}
		
		// Fill _runsConcurrentlyWithReduction for all reduction accesses
		for (std::set<int>::const_iterator it = _reducers.begin(); it != _reducers.end(); ++it) {
			int reducer = *it;
			TaskVerifier *reducerVerifier = (*_verifiers)[reducer];
			assert(reducerVerifier != 0);
			
			for (std::set<int>::const_iterator it2 = _reducers.begin(); it2 != _reducers.end(); ++it2) {
				int other = *it2;
				if (other != reducer && (reducerVerifier->_runsConcurrentlyWith.find(other) == reducerVerifier->_runsConcurrentlyWith.end()))
					reducerVerifier->_runsConcurrentlyWithReduction.insert(other);
			}
			// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
			numTests += reducerVerifier->_runsConcurrentlyWithReduction.size();
#endif
			numTests +=  reducerVerifier->_runsConcurrentlyWithReduction.empty() ? 0 : 1;
		}
	}
	
	void handleReader()
	{
		TaskVerifier *verifier = _verifiers->back();
		assert(verifier != 0);
		assert(verifier->_type == TaskVerifier::READ);
		
		// First reader after writers
		if (_lastAccessType != READERS) {
			flush();
			_lastAccessType = READERS;
		}
		
		// There can only be writers before the reader, unless it's the first access
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
			numTests += _lastWriters.size();
#endif
			numTests += _lastWriters.size();
		}
		
		_lastReaders.insert(verifier->_id);
	}
	
	void handleWriter()
	{
		TaskVerifier *verifier = _verifiers->back();
		assert(verifier != 0);
		assert(verifier->_type == TaskVerifier::WRITE);
		
		flush();
		
		// Writers before writer
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
			numTests += _lastWriters.size();
#endif
			numTests += _lastWriters.size();
		// Readers before writer (either this or previous condition will be
		// true, unless it's the first access)
		} else if (!_lastReaders.empty()) {
			verifier->_runsAfter = _lastReaders;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
#ifdef FINE_SELF_CHECK
			numTests += _lastReaders.size();
#endif
			numTests += _lastReaders.size();
		}
		
		_lastAccessType = WRITER;
		_newWriters.insert(verifier->_id);
	}
	
	void handleReducer()
	{
		TaskVerifier *verifier = _verifiers->back();
		assert(verifier != 0);
		assert((verifier->_type == TaskVerifier::REDUCTION) || (verifier->_type == TaskVerifier::REDUCTION_OTHER));
		
		// First reduction
		if (_lastAccessType != REDUCTION) {
			flush();
			_lastAccessType = REDUCTION;
		}
		
		_newWriters.insert(verifier->_id);
		_reducers.insert(verifier->_id);
	}
	
	void selfcheck() const
	{
#ifdef FINE_SELF_CHECK
#else
		bool globallyValid = true;
#endif
		for (std::vector<TaskVerifier *>::const_iterator vit = _verifiers->begin(); vit != _verifiers->end(); vit++) {
			TaskVerifier *verifier = *vit;
			assert(verifier != 0);
			
			for (std::set<int>::const_iterator it = verifier->_runsAfter.begin(); it != verifier->_runsAfter.end(); it++) {
				int predecessor = *it;
				
				TaskVerifier *predecessorVerifier = (*_verifiers)[predecessor];
				assert(predecessorVerifier != 0);
				assert(predecessorVerifier != verifier);
				
				{
#ifdef FINE_SELF_CHECK
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs after " << predecessorVerifier->_id << " implies " << predecessorVerifier->_id << " runs before " << verifier->_id;
					tap.evaluate(predecessorVerifier->_runsBefore.find(verifier->_id) != predecessorVerifier->_runsBefore.end(), oss.str());
#else
					globallyValid = globallyValid && (predecessorVerifier->_runsBefore.find(verifier->_id) != predecessorVerifier->_runsBefore.end());
#endif
				}
			}
			
			for (std::set<int>::const_iterator it = verifier->_runsBefore.begin(); it != verifier->_runsBefore.end(); it++) {
				int successor = *it;
				
				TaskVerifier *successorVerifier = (*_verifiers)[successor];
				assert(successorVerifier != 0);
				assert(successorVerifier != verifier);
				
				{
#ifdef FINE_SELF_CHECK
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs before " << successorVerifier->_id << " implies " << successorVerifier->_id << " runs after " << verifier->_id;
					tap.evaluate(successorVerifier->_runsAfter.find(verifier->_id) != successorVerifier->_runsAfter.end(), oss.str());
#else
					globallyValid = globallyValid && (successorVerifier->_runsAfter.find(verifier->_id) != successorVerifier->_runsAfter.end());
#endif
				}
			}
			
			for (std::set<int>::const_iterator it = verifier->_runsConcurrentlyWith.begin(); it != verifier->_runsConcurrentlyWith.end(); it++) {
				int concurrent = *it;
				
				TaskVerifier *concurrentVerifier = (*_verifiers)[concurrent];
				assert(concurrentVerifier != 0);
				assert(concurrentVerifier != verifier);
				
				{
#ifdef FINE_SELF_CHECK
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs concurrently with " << concurrentVerifier->_id << " implies " <<
						concurrentVerifier->_id << " runs concurrently with " << verifier->_id;
					tap.evaluate(concurrentVerifier->_runsConcurrentlyWith.find(verifier->_id) != concurrentVerifier->_runsConcurrentlyWith.end(), oss.str());
#else
					globallyValid = globallyValid && (concurrentVerifier->_runsConcurrentlyWith.find(verifier->_id) != concurrentVerifier->_runsConcurrentlyWith.end());
#endif
				}
			}
			
			for (std::set<int>::const_iterator it = verifier->_runsConcurrentlyWithReduction.begin(); it != verifier->_runsConcurrentlyWithReduction.end(); it++) {
				int concurrent = *it;
				
				TaskVerifier *concurrentVerifier = (*_verifiers)[concurrent];
				assert(concurrentVerifier != 0);
				assert(concurrentVerifier != verifier);
				
				{
#ifdef FINE_SELF_CHECK
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs concurrently with " << concurrentVerifier->_id << " implies " <<
						concurrentVerifier->_id << " runs concurrently with " << verifier->_id;
					tap.evaluate(concurrentVerifier->_runsConcurrentlyWithReduction.find(verifier->_id) != concurrentVerifier->_runsConcurrentlyWithReduction.end(), oss.str());
#else
					globallyValid = globallyValid && (concurrentVerifier->_runsConcurrentlyWithReduction.find(verifier->_id) != concurrentVerifier->_runsConcurrentlyWithReduction.end());
#endif
				}
			}
		}
		
#ifdef FINE_SELF_CHECK
#else
		tap.evaluate(globallyValid, "Self verification");
#endif
	}
	
	void submit() const
	{
		for (std::vector<TaskVerifier *>::const_iterator vit = _verifiers->begin(); vit != _verifiers->end(); vit++) {
			TaskVerifier *verifier = *vit;
			assert(verifier != 0);
			verifier->submit(*_verifiers);
		}
	}
	
};


int main(int argc, char **argv)
{
	ncpus = nanos6_get_num_cpus();
	
#if TEST_LESS_THREADS
	ncpus = std::min(ncpus, 64U);
#endif
	
	delayMultiplier = sqrt(ncpus);
	
	if (ncpus < 2) {
		// This test bench only works correctly with at least 2 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test requires at least 2 CPUs");
		tap.end();
		return 0;
	}
	
	std::vector<std::vector<TaskVerifier *> *> testVerifiers;
	std::vector<std::pair<VerifierConstraintCalculator, std::string> > testConstraintCalculators;
	std::vector<Atomic<int> *> testCounters;
	
	int var1;
	
	// Test 1: Reduction before write
	{
		#ifndef FINE_SELF_CHECK
		numTests++;
		#endif
		
		int taskId = 0;
		Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
		Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
		std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
		VerifierConstraintCalculator constraintCalculator(verifiers);
		
		for (long i = 0; i < ncpus - 1; i++) {
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
		}
		
		TaskVerifier *write1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1);
		verifiers->push_back(write1);
		constraintCalculator.handleWriter();
		
		// Forced flush
		constraintCalculator.flush();
		constraintCalculator.flushConcurrent();
		
		testVerifiers.push_back(verifiers);
		testConstraintCalculators.push_back(
				std::make_pair(constraintCalculator, "Subtest 1: Reduction before write"));
		testCounters.push_back(numConcurrentTasks);
		testCounters.push_back(numConcurrentReductionTasks);
	}
	
	// Test 2: Reduction after write
	{
		#ifndef FINE_SELF_CHECK
		numTests++;
		#endif
		
		int taskId = 0;
		Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
		Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
		std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
		VerifierConstraintCalculator constraintCalculator(verifiers);
		
		TaskVerifier *write1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1, numConcurrentTasks);
		verifiers->push_back(write1);
		constraintCalculator.handleWriter();
		
		for (long i = 0; i < ncpus - 1; i++) {
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
		}
		
		// Forced flush
		constraintCalculator.flush();
		constraintCalculator.flushConcurrent();
		
		testVerifiers.push_back(verifiers);
		testConstraintCalculators.push_back(
				std::make_pair(constraintCalculator, "Subtest 2: Reduction after write"));
		testCounters.push_back(numConcurrentTasks);
		testCounters.push_back(numConcurrentReductionTasks);
	}
	
	// Test 3: Reduction before read
	{
		#ifndef FINE_SELF_CHECK
		numTests++;
		#endif
		
		int taskId = 0;
		Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
		Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
		std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
		VerifierConstraintCalculator constraintCalculator(verifiers);
		
		for (long i = 0; i < ncpus - 1; i++) {
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
		}
		
		TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1);
		verifiers->push_back(read1);
		constraintCalculator.handleReader();
		
		// Forced flush
		constraintCalculator.flush();
		constraintCalculator.flushConcurrent();
		
		testVerifiers.push_back(verifiers);
		testConstraintCalculators.push_back(
				std::make_pair(constraintCalculator, "Subtest 3: Reduction before read"));
		testCounters.push_back(numConcurrentTasks);
		testCounters.push_back(numConcurrentReductionTasks);
	}
	
	// Test 4: Reduction after read
	{
		#ifndef FINE_SELF_CHECK
		numTests++;
		#endif
		
		int taskId = 0;
		Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
		Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
		std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
		VerifierConstraintCalculator constraintCalculator(verifiers);
		
		TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1, numConcurrentTasks);
		verifiers->push_back(read1);
		constraintCalculator.handleReader();
		
		for (long i = 0; i < ncpus - 1; i++) {
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
		}
		
		// Forced flush
		constraintCalculator.flush();
		constraintCalculator.flushConcurrent();
		
		testVerifiers.push_back(verifiers);
		testConstraintCalculators.push_back(
				std::make_pair(constraintCalculator, "Subtest 4: Reduction after read"));
		testCounters.push_back(numConcurrentTasks);
		testCounters.push_back(numConcurrentReductionTasks);
	}
	
	// Test 5: Reduction after read after reduction (single)
	{
		// This test only works correctly with at least 3 CPUs
		if (ncpus >= 3) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, 0, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1, numConcurrentTasks);
			verifiers->push_back(read1);
			constraintCalculator.handleReader();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 5: Reduction after read after reduction (single)"));
			testCounters.push_back(numConcurrentTasks);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 6: Reduction after read after reduction (double)
	{
		// This test only works correctly with at least 5 CPUs
		if (ncpus >= 5) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1, numConcurrentTasks2);
			verifiers->push_back(read1);
			constraintCalculator.handleReader();
			
			TaskVerifier *reducer3 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer3);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer4 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer4);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 6: Reduction after read after reduction (double)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 7: Reduction after write after reduction (single)
	{
		// This test only works correctly with at least 3 CPUs
		if (ncpus >= 3) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *write1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1, numConcurrentTasks2);
			verifiers->push_back(write1);
			constraintCalculator.handleWriter();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 7: Reduction after write after reduction (single)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 8: Reduction after write after reduction (double)
	{
		// This test only works correctly with at least 5 CPUs
		if (ncpus >= 5) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			TaskVerifier *write1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1, numConcurrentTasks2);
			verifiers->push_back(write1);
			constraintCalculator.handleWriter();
			
			TaskVerifier *reducer3 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer3);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer4 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer4);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 8: Reduction after write after reduction (double)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 9: Reduction after different reduction (single)
	{
		#ifndef FINE_SELF_CHECK
		numTests++;
		#endif
		
		int taskId = 0;
		Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
		Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
		std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
		VerifierConstraintCalculator constraintCalculator(verifiers);
		
		TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
		verifiers->push_back(reducer1);
		constraintCalculator.handleReducer();
		
		constraintCalculator.flush();
		
		TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks, numConcurrentReductionTasks);
		verifiers->push_back(reducer2);
		constraintCalculator.handleReducer();
		
		// Forced flush
		constraintCalculator.flush();
		constraintCalculator.flushConcurrent();
		
		testVerifiers.push_back(verifiers);
		testConstraintCalculators.push_back(
				std::make_pair(constraintCalculator, "Subtest 9: Reduction after different reduction (single)"));
		testCounters.push_back(numConcurrentTasks);
		testCounters.push_back(numConcurrentReductionTasks);
	}
	
	// Test 10: Reduction after different reduction (double)
	{
		// This test only works correctly with at least 4 CPUs
		if (ncpus >= 4) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			constraintCalculator.flush();
			
			TaskVerifier *reducer3 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer3);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer4 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks, numConcurrentReductionTasks);
			verifiers->push_back(reducer4);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 10: Reduction after different reduction (double)"));
			testCounters.push_back(numConcurrentTasks);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 11: Reduction after write after different reduction (single)
	{
		// This test only works correctly with at least 3 CPUs
		if (ncpus >= 3) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *write1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1, numConcurrentTasks2);
			verifiers->push_back(write1);
			constraintCalculator.handleWriter();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 11: Reduction after write after different reduction (single)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 12: Reduction after write after different reduction (double)
	{
		// This test only works correctly with at least 5 CPUs
		if (ncpus >= 5) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			TaskVerifier *write_1 = new TaskVerifier(taskId, TaskVerifier::WRITE, &var1, numConcurrentTasks2);
			verifiers->push_back(write_1);
			constraintCalculator.handleWriter();
			
			TaskVerifier *reducer3 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer3);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer4 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer4);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 12: Reduction after write after different reduction (double)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 13: Reduction after read after different reduction (single)
	{
		// This test only works correctly with at least 3 CPUs
		if (ncpus >= 3) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1, numConcurrentTasks2);
			verifiers->push_back(read1);
			constraintCalculator.handleReader();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 13: Reduction after read after different reduction (single)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	// Test 14: Reduction after read after different reduction (double)
	{
		// This test only works correctly with at least 5 CPUs
		if (ncpus >= 5) {
			#ifndef FINE_SELF_CHECK
			numTests++;
			#endif
			
			int taskId = 0;
			Atomic<int> *numConcurrentTasks1 = new Atomic<int>(0);
			Atomic<int> *numConcurrentTasks2 = new Atomic<int>(0);
			Atomic<int> *numConcurrentReductionTasks = new Atomic<int>(0);
			std::vector<TaskVerifier *> *verifiers = new std::vector<TaskVerifier *>();
			VerifierConstraintCalculator constraintCalculator(verifiers);
			
			TaskVerifier *reducer1 = new TaskVerifier(taskId, TaskVerifier::REDUCTION, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer1);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer2 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks1, numConcurrentReductionTasks);
			verifiers->push_back(reducer2);
			constraintCalculator.handleReducer();
			
			TaskVerifier *read1 = new TaskVerifier(taskId, TaskVerifier::READ, &var1, numConcurrentTasks2);
			verifiers->push_back(read1);
			constraintCalculator.handleReader();
			
			TaskVerifier *reducer3 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer3);
			constraintCalculator.handleReducer();
			
			TaskVerifier *reducer4 = new TaskVerifier(taskId, TaskVerifier::REDUCTION_OTHER, &var1, numConcurrentTasks2, numConcurrentReductionTasks);
			verifiers->push_back(reducer4);
			constraintCalculator.handleReducer();
			
			// Forced flush
			constraintCalculator.flush();
			constraintCalculator.flushConcurrent();
			
			testVerifiers.push_back(verifiers);
			testConstraintCalculators.push_back(
					std::make_pair(constraintCalculator, "Subtest 14: Reduction after read after different reduction (double)"));
			testCounters.push_back(numConcurrentTasks1);
			testCounters.push_back(numConcurrentTasks2);
			testCounters.push_back(numConcurrentReductionTasks);
		}
	}
	
	tap.registerNewTests(numTests);
	
	tap.begin();
	
	for (std::vector<std::pair<VerifierConstraintCalculator, std::string> >::const_iterator it = testConstraintCalculators.begin();
			it != testConstraintCalculators.end();
			it++)
	{
		tap.emitDiagnostic(it->second);
		
		it->first.selfcheck();
		it->first.submit();
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	for (std::vector<Atomic<int> *>::const_iterator it = testCounters.begin();
			it != testCounters.end();
			it++)
	{
		delete *it;
	}
	
	for (std::vector<std::vector<TaskVerifier *> *>::const_iterator it = testVerifiers.begin();
			it != testVerifiers.end();
			it++)
	{
		std::vector<TaskVerifier *> *verifiers = *it;
		assert(verifiers != 0);
		
		for (std::vector<TaskVerifier *>::const_iterator vit = verifiers->begin();
				vit != verifiers->end();
				vit++)
		{
			TaskVerifier *verifier = *vit;
			assert(verifier != 0);
			delete verifier;
		}
		
		delete verifiers;
	}
	
	return 0;
}
