/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "SortAccessGroups.hpp"

#include "InstrumentGraph.hpp"


namespace Instrument {
	namespace Graph {
		
		
		static void sortAccessGroup(data_access_id_t representantId)
		{
			// Find an order according to the DataAccessRegionIndexer
			DataAccessRegionIndexer<AccessWrapper> accesses;
			{
				data_access_id_t currentId = representantId;
				while (currentId != data_access_id_t()) {
					access_t *current = _accessIdToAccessMap[currentId];
					assert(current != nullptr);
					
					accesses.insert(AccessWrapper(current));
					
					currentId = current->_nextGroupAccess;
				}
			}
			
			{
				access_t *previous = nullptr;
				data_access_id_t newRepresentantId = representantId;
				accesses.processAll(
					[&](DataAccessRegionIndexer<AccessWrapper>::iterator position) -> bool {
						AccessWrapper &wrapper = *position;
						access_t *current = wrapper._access;
						assert(current != nullptr);
						assert(current->_firstGroupAccess == representantId);
						
						if (previous == nullptr) {
							assert(current->_firstGroupAccess == representantId);
							
							newRepresentantId = current->_id;
						} else {
							assert(previous != nullptr);
							assert(previous != current);
							previous->_nextGroupAccess = current->_id;
						}
						current->_firstGroupAccess = newRepresentantId;
						
						previous = current;
						return true;
					}
				);
				
				// There must be at least one!
				assert(previous != nullptr);
				
				previous->_nextGroupAccess = data_access_id_t();
			}
		}
		
		
		static void sortTaskAccessGroups(task_id_t taskId)
		{
			task_info_t &taskInfo = _taskToInfoMap[taskId];
			
			// Sort the task accesses
			taskInfo._liveAccesses.processAll(
				[&](task_live_accesses_t::iterator position) -> bool {
					access_t *access = position->_access;
					assert(access != nullptr);
					
					if (access->_id == access->_firstGroupAccess) {
						sortAccessGroup(access->_id);
					}
					
					return true;
				}
			);
			
			// Clear the list of live accesses since it will be rebuilt with the right regions
			// during replay
			taskInfo._liveAccesses.clear();
			
			for (phase_t *phase : taskInfo._phaseList) {
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(phase);
				
				if (taskGroup == nullptr) {
					continue;
				}
				
				assert(taskGroup != nullptr);
				
				// Sort the fragments if any
				taskGroup->_liveFragments.processAll(
					[&](task_live_access_fragments_t::iterator position) -> bool {
						access_fragment_t *fragment = position->_fragment;
						
						if (fragment->_id == fragment->_firstGroupAccess) {
							sortAccessGroup(fragment->_id);
						}
						
						return true;
					}
				);
				
				// Clear the list of live fragments since it will be rebuilt with the right regions
				// during replay
				taskGroup->_liveFragments.clear();
				
				// Sort the taskwait fragments if any
				taskGroup->_liveTaskwaitFragments.processAll(
					[&](task_live_taskwait_fragments_t::iterator position) -> bool {
						taskwait_fragment_t *fragment = position->_taskwaitFragment;
						
						if (fragment->_id == fragment->_firstGroupAccess) {
							sortAccessGroup(fragment->_id);
						}
						
						return true;
					}
				);
				
				// Clear the list of live taskwait fragments since it will be rebuilt with the right regions
				// during replay
				taskGroup->_liveTaskwaitFragments.clear();
				
				// Recurse to children
				for (task_id_t childId : taskGroup->_children) {
					sortTaskAccessGroups(childId);
				}
			}
		}
		
		
		void sortAccessGroups()
		{
			sortTaskAccessGroups(0);
		}
		
		
		static void clearTaskLiveAccessesAndGroups(task_id_t taskId)
		{
			task_info_t &taskInfo = _taskToInfoMap[taskId];
			
			// Clear the list of live accesses since it will be rebuilt with the right regions
			// during replay
			taskInfo._liveAccesses.clear();
			
			for (phase_t *phase : taskInfo._phaseList) {
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(phase);
				
				if (taskGroup == nullptr) {
					continue;
				}
				
				// Clear the list of live fragments since it will be rebuilt with the right regions
				// during replay
				taskGroup->_liveFragments.clear();
				taskGroup->_liveTaskwaitFragments.clear();
				
				// Recurse to children
				for (task_id_t childId : taskGroup->_children) {
					clearTaskLiveAccessesAndGroups(childId);
				}
			}
		}
		
		
		void clearLiveAccessesAndGroups()
		{
			clearTaskLiveAccessesAndGroups(0);
		}
	}
}

