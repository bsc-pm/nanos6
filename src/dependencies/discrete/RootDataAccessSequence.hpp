/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef ROOT_DATA_ACCESS_SEQUENCE_HPP
#define ROOT_DATA_ACCESS_SEQUENCE_HPP


#include "DataAccessRegion.hpp"
#include "DataAccessSequence.hpp"
#include "RootDataAccessSequenceLinkingArtifacts.hpp"

#include "lowlevel/SpinLock.hpp"


struct RootDataAccessSequence {
	SpinLock _lock;
	RootDataAccessSequenceLinkingArtifacts::hook_type _rootDataAccessSequenceLinks;
	DataAccessSequence _accessSequence;
	
	RootDataAccessSequence()
		: _lock(), _rootDataAccessSequenceLinks(), _accessSequence(&_lock)
	{
	}
	
	RootDataAccessSequence(DataAccessRegion accessRegion)
		: _lock(), _rootDataAccessSequenceLinks(), _accessSequence(accessRegion, &_lock)
	{
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _accessSequence._accessRegion;
	}
};



#endif // ROOT_DATA_ACCESS_SEQUENCE_HPP
