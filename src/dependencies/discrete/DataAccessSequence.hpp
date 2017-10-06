/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_SEQUENCE_HPP
#define DATA_ACCESS_SEQUENCE_HPP

#include <mutex>
#include <boost/intrusive/list.hpp>

#include "DataAccessRegion.hpp"
#include "DataAccessSequenceLinkingArtifacts.hpp"
#include "../DataAccessType.hpp"
#include "lowlevel/SpinLock.hpp"


class Task;
struct DataAccess;


struct DataAccessSequence {
	//! The region of data covered by the accesses of this sequence
	DataAccessRegion _accessRegion;
	
	//! \brief Access originated by the direct parent to the tasks of this access sequence
	DataAccess *_superAccess;
	
	//! \brief SpinLock to protect the sequence and all its subaccesses.
	SpinLock *_lock;
	
	typedef boost::intrusive::list<DataAccess, boost::intrusive::function_hook<DataAccessSequenceLinkingArtifacts>> access_sequence_t;
	
	//! \brief Ordered sequence of accesses to the same location
	access_sequence_t _accessSequence;
	
	
	DataAccessSequence() = delete;
	inline DataAccessSequence(SpinLock *lock);
	inline DataAccessSequence(DataAccessRegion accessRegion, SpinLock *lock);
	inline DataAccessSequence(DataAccessRegion accessRegion, DataAccess *superAccess, SpinLock *lock);
	
	
	//! \brief Get a locking guard
	inline std::unique_lock<SpinLock> getLockGuard();
	
	
	//! \brief Get the Effective Previous access of another given one
	//! 
	//! \param[in] dataAccess the DataAccess that is effectively after the one to be returned or nullptr if the DataAccess is yet to be added and the sequence is empty
	//! 
	//! \returns the Effective Previous access to the one passed by parameter, or nullptr if there is none
	//! 
	//! NOTE: This function assumes that the whole hierarchy has already been locked
	inline DataAccess *getEffectivePrevious(DataAccess *dataAccess);
};


#include "DataAccess.hpp"
#include "DataAccessSequenceImplementation.hpp"
#include "DataAccessSequenceLinkingArtifactsImplementation.hpp"


#endif // DATA_ACCESS_SEQUENCE_HPP
