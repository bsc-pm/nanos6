#include <atomic>
#include <cassert>
#include <set>
#include <vector>

#include <math.h>

#include "TestAnyProtocolProducer.hpp"

#include <nanos6/debug.h>


#define SUSTAIN_MICROSECONDS 200000L


TestAnyProtocolProducer tap;

static int nextTaskId = 0;

static int numTests = 0;

static unsigned int ncpus = 0;
static double delayMultiplier = 1.0;


struct TaskVerifier;
std::vector<TaskVerifier *> verifiers;


void shutdownTests()
{
}


struct TaskVerifier {
	typedef enum {
		READ,
		WRITE,
		CONCURRENT,
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
	status_t _status;
	type_t _type;
	int *_variable;
	std::atomic_int *_numConcurrentTasks;
	
	TaskVerifier() = delete;
	
	TaskVerifier(type_t type, int *variable, std::atomic_int *numConcurrentTasks = nullptr)
		: _id(nextTaskId++), _runsAfter(), _runsBefore(), _runsConcurrentlyWith(), _status(NOT_STARTED)
		, _type(type), _variable(variable), _numConcurrentTasks(numConcurrentTasks)
	{
	}
	
	
	char const *type2String() const
	{
		switch (_type) {
			case READ:
				return "READ";
			case WRITE:
				return "WRITE";
			case CONCURRENT:
				return "CONCURRENT";
			case REDUCTION:
				return "REDUCTION";
			case REDUCTION_OTHER:
				return "REDUCTION OTHER";
		}
	}
	
	void submit();
	
	void verify()
	{
		assert(_status == NOT_STARTED);
		tap.emitDiagnostic("Task ", _id, " (", type2String(), ") starts");
		_status = STARTED;
		
		for (int predecessor : _runsAfter) {
			TaskVerifier *predecessorTask = verifiers[predecessor];
			assert(predecessorTask != 0);
			{
				std::ostringstream oss;
				oss << "Task " << _id << " must run after task " << predecessorTask->_id;
				tap.evaluate(predecessorTask->_status == FINISHED, oss.str());
			}
		}
		
		if (!_runsConcurrentlyWith.empty()) {
			// FIXME can be extended to a full wait when taskyield is implemented
			int nwait = (ncpus < _runsConcurrentlyWith.size() + 1) ? ncpus : _runsConcurrentlyWith.size() + 1;
			
			assert(_numConcurrentTasks != nullptr);
			int var = ++(*_numConcurrentTasks);
			tap.emitDiagnostic("Task ", var, "/", nwait, ", running concurrently within its group, enters synchronization");
			
			std::ostringstream oss;
			oss << "Task " << _id << " can run concurrently with other tasks filling up the number of available CPUs";
			tap.timedEvaluate(
				[&]() {
					var = _numConcurrentTasks->load();
					return var >= nwait;
				},
				SUSTAIN_MICROSECONDS * delayMultiplier,
				oss.str()
			);
		}
		
		struct timespec delay = { 0, 1000000};
		nanosleep(&delay, &delay);
		
		for (int successor : _runsBefore) {
			TaskVerifier *successorTask = verifiers[successor];
			assert(successorTask != 0);
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


#pragma omp task in(*variable) label(R)
void verifyRead(int *variable, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task out(*variable) label(W)
void verifyWrite(int *variable, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task concurrent(*variable1) label(C)
void verifyConcurrent(int *variable1, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


void verifyReduction(int *variable1, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


void TaskVerifier::submit()
{
	switch (_type) {
		case READ:
			verifyRead(_variable, this);
			break;
		case WRITE:
			verifyWrite(_variable, this);
			break;
		case CONCURRENT:
			verifyConcurrent(_variable, this);
			break;
		case REDUCTION: {
			int& red_variable = *_variable;
			#pragma omp task reduction(+: red_variable) label(RED)
			verifyReduction(_variable, this);
			break;
		}
		case REDUCTION_OTHER: {
			int& red_variable = *_variable;
			#pragma omp task reduction(*: red_variable) label(RED_OTHER)
			verifyReduction(_variable, this);
			break;
		}
	}
}


struct VerifierConstraintCalculator {
	typedef enum {
		READERS,
		WRITER,
		CONCURRENT,
		REDUCTION
	} access_type_t;
	
	access_type_t _lastAccessType;
	
	std::set<int> _lastWriters;
	std::set<int> _lastReaders;
	std::set<int> _newWriters;
	
	VerifierConstraintCalculator()
		: _lastAccessType(READERS), _lastWriters(), _lastReaders(), _newWriters()
	{}
	
	// Fills out the _runsBefore and _runsConcurrentlyWith members of the verifier that is about to exit the current view of the status
	void flush()
	{
		if (_lastAccessType == READERS) {
			// There can only be writers before last access, unless it's the first access
			for (int writer : _lastWriters) {
				TaskVerifier *writerVerifier = verifiers[writer];
				assert(writerVerifier != 0);
				
				writerVerifier->_runsBefore = _lastReaders;
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += _lastReaders.size()*2;
				
				for (int other : _lastWriters) {
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += writerVerifier->_runsConcurrentlyWith.size();
				numTests +=  writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastWriters.clear();
		} else {
			assert(_lastAccessType == WRITER || _lastAccessType == CONCURRENT || _lastAccessType == REDUCTION);
			
			// Readers before last access
			for (int reader : _lastReaders) {
				TaskVerifier *readerVerifier = verifiers[reader];
				assert(readerVerifier != 0);
				
				readerVerifier->_runsBefore = _newWriters;
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += _newWriters.size()*2;
				
				for (int other : _lastReaders) {
					if (other != reader)
						readerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += readerVerifier->_runsConcurrentlyWith.size();
				numTests += readerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastReaders.clear();
			
			// Writer(s) before last access (either this or previous set will
			// be non-empty, but not both unless it's the first access)
			for (int writer : _lastWriters) {
				TaskVerifier *writerVerifier = verifiers[writer];
				assert(writerVerifier != 0);
				
				writerVerifier->_runsBefore = _newWriters;
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += _newWriters.size()*2;
				
				for (int other : _lastWriters) {
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += writerVerifier->_runsConcurrentlyWith.size();
				numTests += writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
			_lastWriters = _newWriters;
			_newWriters.clear();
		}
	}
	
	// Fills out the _runsConcurrentlyWith member of the very last group of accesses
	void flushConcurrent()
	{
		if (_lastAccessType == READERS) {
			for (int reader : _lastReaders) {
				TaskVerifier *readerVerifier = verifiers[reader];
				assert(readerVerifier != 0);
				
				for (int other : _lastReaders) {
					if (other != reader)
						readerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += readerVerifier->_runsConcurrentlyWith.size();
				numTests += readerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
		} else {
			assert(_lastAccessType == WRITER || _lastAccessType == CONCURRENT || _lastAccessType == REDUCTION);
			
			for (int writer : _lastWriters) {
				TaskVerifier *writerVerifier = verifiers[writer];
				assert(writerVerifier != 0);
				
				for (int other : _lastWriters) {
					if (other != writer)
						writerVerifier->_runsConcurrentlyWith.insert(other);
				}
				// Increment number of tests, corresponding to tests run by selfcheck and verify
				numTests += writerVerifier->_runsConcurrentlyWith.size();
				numTests +=  writerVerifier->_runsConcurrentlyWith.empty() ? 0 : 1;
			}
		}
	}
	
	void handleReader(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		// First reader after writers
		if (_lastAccessType != READERS) {
			flush();
			_lastAccessType = READERS;
		}
		
		// There can only be writers before the reader, unless it's the first access
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastWriters.size()*2;
		}
		
		_lastReaders.insert(verifier->_id);
	}
	
	void handleWriter(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		flush();
		
		// Writers before writer
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastWriters.size()*2;
		// Readers before writer (either this or previous condition will be
		// true, unless it's the first access)
		} else if (!_lastReaders.empty()) {
			verifier->_runsAfter = _lastReaders;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastReaders.size()*2;
		}
		
		_lastAccessType = WRITER;
		_newWriters.insert(verifier->_id);
	}
	
	void handleConcurrent(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		// First concurrent
		if (_lastAccessType != CONCURRENT) {
			flush();
			_lastAccessType = CONCURRENT;
		}
		
		// Writer(s) before writers
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastWriters.size()*2;
		// Readers before writers (either this or previous condition will be
		// true, unless it's the first access)
		} else if (!_lastReaders.empty()) {
			verifier->_runsAfter = _lastReaders;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastReaders.size()*2;
		}
		
		_newWriters.insert(verifier->_id);
	}
	
	void handleReducer(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		// First reduction
		if (_lastAccessType != REDUCTION) {
			flush();
			_lastAccessType = REDUCTION;
		}
		
		// Writer(s) before writers
		if (!_lastWriters.empty()) {
			verifier->_runsAfter = _lastWriters;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastWriters.size()*2;
		// Readers before writers (either this or previous condition will be
		// true, unless it's the first access)
		} else if (!_lastReaders.empty()) {
			verifier->_runsAfter = _lastReaders;
			// Increment number of tests, corresponding to tests run by selfcheck and verify
			numTests += _lastReaders.size()*2;
		}
		
		_newWriters.insert(verifier->_id);
	}
	
	static void selfcheck()
	{
		for (TaskVerifier *verifier : verifiers) {
			assert(verifier != 0);
			
			for (int predecessor : verifier->_runsAfter) {
				TaskVerifier *predecessorVerifier = verifiers[predecessor];
				assert(predecessorVerifier != 0);
				
				{
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs after " << predecessorVerifier->_id << " implies " << predecessorVerifier->_id << " runs before " << verifier->_id;
					tap.evaluate(predecessorVerifier->_runsBefore.find(verifier->_id) != predecessorVerifier->_runsBefore.end(), oss.str());
				}
			}
			
			for (int successor : verifier->_runsBefore) {
				TaskVerifier *successorVerifier = verifiers[successor];
				assert(successorVerifier != 0);
				
				{
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs before " << successorVerifier->_id << " implies " << successorVerifier->_id << " runs after " << verifier->_id;
					tap.evaluate(successorVerifier->_runsAfter.find(verifier->_id) != successorVerifier->_runsAfter.end(), oss.str());
				}
			}
			
			for (int concurrent : verifier->_runsConcurrentlyWith) {
				TaskVerifier *concurrentVerifier = verifiers[concurrent];
				assert(concurrentVerifier != 0);
				
				{
					std::ostringstream oss;
					oss << "Self verification: " << verifier->_id << " runs concurrently with " << concurrentVerifier->_id << " implies " <<
						concurrentVerifier->_id << " runs concurrently with " << verifier->_id;
					tap.evaluate(concurrentVerifier->_runsConcurrentlyWith.find(verifier->_id) != concurrentVerifier->_runsConcurrentlyWith.end(), oss.str());
				}
			}
		}
	}
	
};


static VerifierConstraintCalculator _constraintCalculator;


int main(int argc, char **argv)
{
	ncpus = nanos_get_num_cpus();
	delayMultiplier = sqrt(ncpus);
	
	int var1;
	
	// 1 writer
	TaskVerifier firstWriter(TaskVerifier::WRITE, &var1); verifiers.push_back(&firstWriter); _constraintCalculator.handleWriter(&firstWriter);
	
	// NCPUS reducers
	std::atomic_int numReducers1(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reducer = new TaskVerifier(TaskVerifier::REDUCTION, &var1, &numReducers1);
		verifiers.push_back(reducer);
		_constraintCalculator.handleReducer(reducer);
	}
	
	// NCPUS readers
	std::atomic_int numConcurrentReaders1(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reader = new TaskVerifier(TaskVerifier::READ, &var1, &numConcurrentReaders1);
		verifiers.push_back(reader);
		_constraintCalculator.handleReader(reader);
	}
	
	// NCPUS reducers
	std::atomic_int numReducers2(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reducer = new TaskVerifier(TaskVerifier::REDUCTION, &var1, &numReducers2);
		verifiers.push_back(reducer);
		_constraintCalculator.handleReducer(reducer);
	}

	// NCPUS concurrent
	std::atomic_int numConcurrents1(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *concurrent = new TaskVerifier(TaskVerifier::CONCURRENT, &var1, &numConcurrents1);
		verifiers.push_back(concurrent);
		_constraintCalculator.handleConcurrent(concurrent);
	}
	
	// NCPUS reducers
	std::atomic_int numReducers3(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reducer = new TaskVerifier(TaskVerifier::REDUCTION, &var1, &numReducers3);
		verifiers.push_back(reducer);
		_constraintCalculator.handleReducer(reducer);
	}
	
	// NCPUS reducers (different operation)
	_constraintCalculator.flush();
	std::atomic_int numReducers4(0);
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reducer = new TaskVerifier(TaskVerifier::REDUCTION_OTHER, &var1, &numReducers4);
		verifiers.push_back(reducer);
		_constraintCalculator.handleReducer(reducer);
	}
	
	// Forced flush
	_constraintCalculator.flush();
	_constraintCalculator.flushConcurrent();
	
	tap.registerNewTests(numTests);
	tap.begin();
	
	_constraintCalculator.selfcheck();
	
	for (TaskVerifier *verifier : verifiers) {
		assert(verifier != 0);
		verifier->submit();
	}
	
	#pragma omp taskwait
	
	tap.end();
	
	return 0;
}
