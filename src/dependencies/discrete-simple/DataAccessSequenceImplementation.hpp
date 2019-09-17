#ifndef DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
#define DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP

#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

void DataAccessSequence::removeCompleteSuccessors(DataAccessType type)
{
	assert(!_sequence.empty());
	
	if (type == READ_ACCESS_TYPE) {
#ifndef NDEBUG
		unsigned int completedReaders = 0;
#endif
		sequence_t::iterator it;
		for (it = _sequence.begin(); it != _sequence.end(); ++it) {
			if (it->isWriter()) break;
#ifndef NDEBUG
			++completedReaders;
#endif
		}
		assert(completedReaders > 0);
		assert(completedReaders == _satisfiedReaders);
		
		_sequence.erase(_sequence.begin(), it);
	} else {
		assert(_sequence.front().isWriter());
		//_sequence.pop_front();
        _sequence.erase(_sequence.begin());
	}
	
	_satisfiedReaders = 0;
	_uncompletedReaders = 0;
	_uncompletedWriters = 0;
}

void DataAccessSequence::satisfyOldestSuccessors(satisfied_originator_list_t &satisfiedOriginators)
{
	bool topmost = true;
	sequence_t::iterator it;
	for (it = _sequence.begin(); it != _sequence.end(); ++it) {
		DataAccess &dataAccess = *it;
		
		if (dataAccess.isWriter() && !topmost) {
			break;
		}
		
		Task *successor = dataAccess.getOriginator();
		assert(successor != nullptr);
		if (successor->decreasePredecessors()) {
			satisfiedOriginators.push_back(successor);
		}
		
		if (dataAccess.isWriter()) {
			++_uncompletedWriters;
			break;
		}
		
		topmost = false;
		++_satisfiedReaders;
		++_uncompletedReaders;
	}
}

#endif // DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
