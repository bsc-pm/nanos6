/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMMUTATIVE_SCOREBOARD_HPP
#define COMMUTATIVE_SCOREBOARD_HPP


#include <DataAccessRegion.hpp>
#include <LinearRegionMap.hpp>
#include <lowlevel/PaddedTicketSpinLock.hpp>

#include <set>

class Task;
struct CPUDependencyData;


struct CommutativeScoreboard {
	struct entry_t {
		DataAccessRegion _region;
		bool _available;
		std::set<Task *> _participants;
		
		entry_t(DataAccessRegion const &region)
			: _region(region), _available(true)
		{
		}
		
		DataAccessRegion const &getAccessRegion() const
		{
			return _region;
		}
		DataAccessRegion &getAccessRegion()
		{
			return _region;
		}
	};
	
	typedef PaddedTicketSpinLock<> lock_t;
	typedef LinearRegionMap<entry_t> map_t;
	
	static lock_t _lock;
	static map_t _map;
	
	static bool addAndEvaluateTask(Task *task, CPUDependencyData &hpDependencyData);
	static void processReleasedCommutativeRegions(CPUDependencyData &hpDependencyData);
	
private:
	typedef std::set<Task *> candidate_set_t;
	
	static inline bool acquireEntry(entry_t &entry);
	static inline void evaluateCompetingTask(Task *task, CPUDependencyData &hpDependencyData, candidate_set_t &candidates);
};


#endif // COMMUTATIVE_SCOREBOARD_HPP
