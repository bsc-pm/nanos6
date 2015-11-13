#include <atomic>
#include <cassert>
#include <cstdio>
#include <list>
#include <set>
#include <sstream>
#include <vector>

#include <time.h>
#include <string.h>
#include <unistd.h>

#include "infrastructure/ProgramLifecycle.hpp"
#include "infrastructure/TestAnyProtocolProducer.hpp"
#include "infrastructure/Timer.hpp"

#include <nanos6_debug_interface.h>


extern TestAnyProtocolProducer tap;
std::atomic<bool> theOuterTaskHasFinished;


void shutdownTests()
{
}



class OuterExplicitTaskChecker {
public:
	std::atomic<bool> _mainHasFinished;
	
	OuterExplicitTaskChecker():
		_mainHasFinished(false)
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


static int nextTaskId = 0;

struct TaskVerifier;
std::vector<TaskVerifier *> verifiers;


static int numTests = 0;


struct TaskVerifier {
	typedef enum {
		READ,
		WRITE,
		DOUBLE_READ,
		DOUBLE_WRITE,
		UPGRADE1,
		UPGRADE2,
		UPGRADE3
	} type_t;
	
	typedef enum {
		NOT_STARTED,
		STARTED,
		FINISHED
	} status_t;
	
	int _id;
	std::set<int> _runsAfter;
	std::set<int> _runsBefore;
	status_t _status;
	type_t _type;
	int *_variable;
	
	TaskVerifier() = delete;
	
	TaskVerifier(type_t type, int *variable)
		: _id(nextTaskId++), _runsAfter(), _runsBefore(), _status(NOT_STARTED)
		, _type(type), _variable(variable)
	{
	}
	
	
	char const *type2String() const
	{
		switch (_type) {
			case READ:
				return "READ";
			case WRITE:
				return "WRITE";
			case DOUBLE_READ:
				return "READ READ";
			case DOUBLE_WRITE:
				return "WRITE WRITE";
			case UPGRADE1:
				return "READ WRITE";
			case UPGRADE2:
				return "WRITE READ";
			case UPGRADE3:
				return "READ WRITE READ WRITE";
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



#pragma omp task in(*variable, *verifier) label(R)
void verifyRead(int *variable, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}

#pragma omp task out(*variable) in(*verifier) label(W)
void verifyWrite(int *variable, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task in(*variable1, *variable2, *verifier) label(RR)
void verifyRepeatedRead(int *variable1, int *variable2, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task out(*variable1, *variable2) in(*verifier) label(WW)
void verifyRepeatedWrite(int *variable1, int *variable2, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task out(*variable2) in(*variable1, *verifier) label(RW)
void verifyUpgradedAccess1(int *variable1, int *variable2, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task out(*variable1) in(*variable2, *verifier) label(WR)
void verifyUpgradedAccess2(int *variable1, int *variable2, TaskVerifier *verifier)
{
	assert(verifier != 0);
	verifier->verify();
}


#pragma omp task out(*variable2, *variable4) in(*variable1, *variable3, *verifier) label(RWRW)
void verifyUpgradedAccess3(int *variable1, int *variable2, int *variable3, int *variable4, TaskVerifier *verifier)
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
		case DOUBLE_READ:
			verifyRepeatedRead(_variable, _variable, this);
			break;
		case DOUBLE_WRITE:
			verifyRepeatedWrite(_variable, _variable, this);
			break;
		case UPGRADE1:
			verifyUpgradedAccess1(_variable, _variable, this);
			break;
		case UPGRADE2:
			verifyUpgradedAccess2(_variable, _variable, this);
			break;
		case UPGRADE3:
			verifyUpgradedAccess3(_variable, _variable, _variable, _variable, this);
			break;
	}
}


struct VerifierConstraintCalculator {
	typedef enum {
		READERS,
		WRITER
	} access_type_t;
	
	access_type_t _lastAccessType;
	
	int _lastWriter;
	std::set<int> _lastReaders;
	
	VerifierConstraintCalculator()
		: _lastAccessType(READERS), _lastWriter(-1), _lastReaders()
	{}
	
	// Fills out the _runsBefore member of the verifier that is about to exit the current view of the status
	void flush(TaskVerifier *newWriteVerifier = 0)
	{
		if (_lastAccessType == READERS) {
			if (_lastWriter != -1) {
				TaskVerifier *lastWriter = verifiers[_lastWriter];
				assert(lastWriter != 0);
				
				lastWriter->_runsBefore = _lastReaders;
				numTests += _lastReaders.size();
				_lastWriter = -1;
			}
		} else {
			assert(_lastAccessType == WRITER);
			
			if (_lastWriter != -1) {
				for (int reader : _lastReaders) {
					TaskVerifier *readerVerifyer = verifiers[reader];
					assert(readerVerifyer != 0);
					
					readerVerifyer->_runsBefore.insert(_lastWriter);
					numTests++;
				}
				_lastReaders.clear();
				
				if (newWriteVerifier != 0) {
					TaskVerifier *lastWriter = verifiers[_lastWriter];
					assert(lastWriter != 0);
					
					lastWriter->_runsBefore.insert(newWriteVerifier->_id);
					numTests++;
				}
			}
		}
	}
	
	void handleReader(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		if (_lastAccessType != READERS) {
			flush();
			_lastAccessType = READERS;
		}
		
		if (_lastWriter != -1) {
			verifier->_runsAfter.insert(_lastWriter);
			numTests++;
		}
		
		_lastReaders.insert(verifier->_id);
	}
	
	void handleWriter(TaskVerifier *verifier)
	{
		assert(verifier != 0);
		
		flush(verifier);
		if (_lastWriter != -1) {
			verifier->_runsAfter.insert(_lastWriter);
			numTests++;
		}
		for (int reader : _lastReaders) {
			verifier->_runsAfter.insert(reader);
			numTests++;
		}
		
		_lastAccessType = WRITER;
		_lastWriter = verifier->_id;
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
		}
	}
	
};


static VerifierConstraintCalculator _constraintCalculator;




int main(int argc, char **argv)
{
	initializationTimer.stop();
	
	long ncpus = nanos_get_num_cpus();
	
	int var1;
	
	// 1 writer
	TaskVerifier firstWriter(TaskVerifier::WRITE, &var1); verifiers.push_back(&firstWriter); _constraintCalculator.handleWriter(&firstWriter);
	
	// NCPUS readers
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reader = new TaskVerifier(TaskVerifier::READ, &var1);
		verifiers.push_back(reader);
		_constraintCalculator.handleReader(reader);
	}
	
	// 1 writer
	TaskVerifier secondWriter(TaskVerifier::WRITE, &var1); verifiers.push_back(&secondWriter); _constraintCalculator.handleWriter(&secondWriter);
	
	// 1 writer
	TaskVerifier thirdWriter(TaskVerifier::WRITE, &var1); verifiers.push_back(&thirdWriter); _constraintCalculator.handleWriter(&thirdWriter);
	
	// NCPUS readers
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reader = new TaskVerifier(TaskVerifier::READ, &var1);
		verifiers.push_back(reader);
		_constraintCalculator.handleReader(reader);
	}
	
	// 1 double writer
	TaskVerifier firstRewriter(TaskVerifier::DOUBLE_WRITE, &var1); verifiers.push_back(&firstRewriter); _constraintCalculator.handleWriter(&firstRewriter);
	
	// NCPUS double readers
	for (long i=0; i < ncpus; i++) {
		TaskVerifier *reader = new TaskVerifier(TaskVerifier::DOUBLE_READ, &var1);
		verifiers.push_back(reader);
		_constraintCalculator.handleReader(reader);
	}
	
	// 1 double writer
	TaskVerifier secondRewriter(TaskVerifier::DOUBLE_WRITE, &var1); verifiers.push_back(&secondRewriter); _constraintCalculator.handleWriter(&secondRewriter);
	
	// 1 double writer
	TaskVerifier thirdRewriter(TaskVerifier::DOUBLE_WRITE, &var1); verifiers.push_back(&thirdRewriter); _constraintCalculator.handleWriter(&thirdRewriter);
	
	// Upgraded access form 1
	TaskVerifier upgradedForm1(TaskVerifier::UPGRADE1, &var1); verifiers.push_back(&upgradedForm1); _constraintCalculator.handleWriter(&upgradedForm1);
	
	// Upgraded access form 2
	TaskVerifier upgradedForm2(TaskVerifier::UPGRADE2, &var1); verifiers.push_back(&upgradedForm2); _constraintCalculator.handleWriter(&upgradedForm2);
	
	// Upgraded access form 3
	TaskVerifier upgradedForm3(TaskVerifier::UPGRADE3, &var1); verifiers.push_back(&upgradedForm3); _constraintCalculator.handleWriter(&upgradedForm3);
	
	_constraintCalculator.flush();
	
	tap.registerNewTests(numTests * 2);
	tap.begin();
	
	_constraintCalculator.selfcheck();
	
	for (TaskVerifier *verifier : verifiers) {
		assert(verifier != 0);
		verifier->submit();
	}
	
	#pragma omp taskwait
	
	shutdownTimer.start();
	
	tap.end();
	
	return 0;
}

