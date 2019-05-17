/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <sstream>
#include <string>

#include "ExecutionSteps.hpp"
#include "InstrumentComputePlaceId.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskId.hpp"


namespace Instrument {
	namespace Graph {
		void execution_step_t::emitCPUAndTask(std::ostringstream & oss)
		{
			if (_instrumentationContext._externalThreadName != nullptr) {
				oss << "External Thread " << *_instrumentationContext._externalThreadName << " task " << _instrumentationContext._taskId;
			} else {
				oss << "CPU " << _instrumentationContext._computePlaceId;
				if (_instrumentationContext._taskId != task_id_t()) {
					oss << " task " << _instrumentationContext._taskId;
				}
			}
		}
		
		void execution_step_t::emitCPU(std::ostringstream & oss)
		{
			if (_instrumentationContext._externalThreadName != nullptr) {
				oss << "External Thread " << *_instrumentationContext._externalThreadName;
			} else {
				oss << "CPU " << _instrumentationContext._computePlaceId;
			}
		}
		
		
		void create_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_newTaskId];
			assert(taskInfo._status == not_created_status);
			
			taskInfo._status = not_started_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string create_task_step_t::describe()
		{
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": start creating task " << _newTaskId;
			return oss.str();
		}
		
		
		bool create_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void enter_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert((taskInfo._status == not_started_status) || (taskInfo._status == blocked_status));
			
			taskInfo._status = started_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string enter_task_step_t::describe()
		{
			std::ostringstream oss;
			emitCPU(oss);
			oss << ": enter task " << _instrumentationContext._taskId;
			return oss.str();
		}
		
		
		bool enter_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void exit_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = finished_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string exit_task_step_t::describe()
		{
			std::ostringstream oss;
			emitCPU(oss);
			oss << ": exit task " << _instrumentationContext._taskId;
			return oss.str();
		}
		
		
		bool exit_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void enter_taskwait_step_t::execute()
		{
			if (_instrumentationContext._taskId == task_id_t()) {
				return;
			}
			
			taskwait_t *taskwait = _taskwaitToInfoMap[_taskwaitId];
			assert(taskwait != nullptr);
			
			taskwait->_status = started_status;
			taskwait->_lastCPU = _instrumentationContext._computePlaceId;
			
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = blocked_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string enter_taskwait_step_t::describe()
		{
			if (_instrumentationContext._taskId == task_id_t()) {
				return "An external thread enters a taskwait";
			}
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": enter taskwait " << _taskwaitId;
			return oss.str();
		}
		
		
		bool enter_taskwait_step_t::visible()
		{
			return (_instrumentationContext._taskId != task_id_t());
		}
		
		
		
		void exit_taskwait_step_t::execute()
		{
			if (_instrumentationContext._taskId == task_id_t()) {
				return;
			}
			
			taskwait_t *taskwait = _taskwaitToInfoMap[_taskwaitId];
			assert(taskwait != nullptr);
			
			taskwait->_status = finished_status;
			taskwait->_lastCPU =_instrumentationContext._computePlaceId;
			
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert(taskInfo._status == blocked_status);
			taskInfo._status = started_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string exit_taskwait_step_t::describe()
		{
			if (_instrumentationContext._taskId == task_id_t()) {
				return "An external thread exits a taskwait";
			}
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": exit taskwait " << _taskwaitId;
			return oss.str();
		}
		
		
		bool exit_taskwait_step_t::visible()
		{
			return (_instrumentationContext._taskId != task_id_t());
		}
		
		
		
		void enter_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert((taskInfo._status == started_status) || (taskInfo._status == blocked_status));
			
			taskInfo._status = started_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string enter_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": enter usermutex " << _usermutexId;
			return oss.str();
		}
		
		
		bool enter_usermutex_step_t::visible()
		{
			return true;
		}
		
		
		
		void block_on_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = blocked_status;
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string block_on_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": blocks on usermutex " << _usermutexId;
			return oss.str();
		}
		
		
		bool block_on_usermutex_step_t::visible()
		{
			return true;
		}
		
		
		
		void exit_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_instrumentationContext._taskId];
			assert(taskInfo._status == started_status);
			
			if (taskInfo._lastCPU == _instrumentationContext._computePlaceId) {
				// Not doing anything for now
				// Perhaps will represent the state of the mutex, its allocation slots,
				// and links from those to task-internal critical nodes
			}
			taskInfo._lastCPU = _instrumentationContext._computePlaceId;
		}
		
		
		std::string exit_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": exit usermutex " << _usermutexId;
			return oss.str();
		}
		
		
		bool exit_usermutex_step_t::visible()
		{
			return true;
		}
		
		
		
		void create_data_access_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_superAccess = _superAccessId;
			access->_originator = _originatorTaskId;
			access->_type = _accessType;
			access->_accessRegion = _region;
			access->weak() = _weak;
			access->satisfied() = _globallySatisfied;
			access->_status = created_access_status;
			
			task_info_t &taskInfo = _taskToInfoMap[_originatorTaskId];
			
			if ((access->_objectType == regular_access_type) || (access->_objectType == entry_fragment_type)) {
				assert(taskInfo._parent != task_id_t());
				assert(!taskInfo._liveAccesses.contains(_region));
				taskInfo._liveAccesses.insert(AccessWrapper(access));
			} else {
				task_group_t *taskGroup = dynamic_cast<task_group_t *> (taskInfo._phaseList[access->_parentPhase]);
				assert(taskGroup != nullptr);
				
				assert(!taskGroup->_liveTaskwaitFragments.contains(_region));
				taskGroup->_liveTaskwaitFragments.insert(TaskwaitFragmentWrapper((taskwait_fragment_t *) access));
			}
		}
		
		
		std::string create_data_access_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": creates";
			if (_globallySatisfied) {
				oss << " satisfied";
			}
			if (_readSatisfied) {
				oss << " readSatisfied";
			}
			if (_readSatisfied) {
				oss << " writeSatisfied";
			}
			
			if (_weak) {
				oss << " weak";
			}
			switch (_accessType) {
				case READ_ACCESS_TYPE:
					oss << " R";
					break;
				case WRITE_ACCESS_TYPE:
					oss << " W";
					break;
				case READWRITE_ACCESS_TYPE:
					oss << " RW";
					break;
				case CONCURRENT_ACCESS_TYPE:
					oss << " CRR";
					break;
				case REDUCTION_ACCESS_TYPE:
					oss << " RED";
					break;
				case COMMUTATIVE_ACCESS_TYPE:
					oss << " CMM";
					break;
				case NO_ACCESS_TYPE:
					oss << " LOC";
					break;
			}
			
			oss << " ";
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId << " [" << _region << "]";
			
			return oss.str();
		}
		
		
		bool create_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void upgrade_data_access_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_type = _newAccessType;
			if (_becomesUnsatisfied) {
				access->satisfied() = false;
			}
		}
		
		
		std::string upgrade_data_access_step_t::describe()
		{
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": upgrades access " << _accessId << " to";
			if (_becomesUnsatisfied) {
				oss << " unsatisfied";
			}
			if (_newWeakness) {
				oss << " weak";
			} else {
				oss << " strong";
			}
			switch (_newAccessType) {
				case READ_ACCESS_TYPE:
					oss << " R";
					break;
				case WRITE_ACCESS_TYPE:
					oss << " W";
					break;
				case READWRITE_ACCESS_TYPE:
					oss << " RW";
					break;
				case CONCURRENT_ACCESS_TYPE:
					oss << " CRR";
					break;
				case REDUCTION_ACCESS_TYPE:
					oss << " RED";
					break;
				case COMMUTATIVE_ACCESS_TYPE:
					oss << " CMM";
					break;
				case NO_ACCESS_TYPE:
					oss << " LOC";
					break;
			}
			return oss.str();
		}
		
		
		bool upgrade_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void data_access_becomes_satisfied_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->satisfied() = access->satisfied() | _globallySatisfied;
		}
		
		
		std::string data_access_becomes_satisfied_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": makes";
			
			oss << " ";
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			oss << " ";
			
			oss << _accessId << " of task " << _targetTaskId;
			if (_globallySatisfied) {
				oss << " satisfied";
			}
			return oss.str();
		}
		
		
		bool data_access_becomes_satisfied_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void modified_data_access_region_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_accessRegion = _region;
			
// 			if (access->fragment()) {
// 				access_fragment_t *fragment = (access_fragment_t *) access;
// 				
// 				task_group_t *taskGroup = fragment->_taskGroup;
// 				assert(taskGroup != nullptr);
// 				
// 				taskGroup->_fragments.moved(fragment);
// 			} else {
// 				task_info_t &taskInfo = _taskToInfoMap[access->_originator];
// 				taskInfo._accesses.moved(access);
// 			}
		}
		
		
		std::string modified_data_access_region_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": changes region of ";
			
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId << " to [" << _region << "]";
			
			return oss.str();
		}
		
		
		bool modified_data_access_region_step_t::visible()
		{
			return _showDependencyStructures && _showRegions;
		}
		
		
		
		void fragment_data_access_step_t::execute()
		{
			access_t *original = _accessIdToAccessMap[_originalAccessId];
			access_t *newFragment = _accessIdToAccessMap[_newFragmentAccessId];
			assert(original != nullptr);
			assert(newFragment != nullptr);
			
			newFragment->_type = original->_type;
			newFragment->_accessRegion = _newRegion;
			newFragment->_bitset = original->_bitset;
			newFragment->_status = original->_status;
			newFragment->_otherProperties = original->_otherProperties;
			
			if (newFragment->_objectType == entry_fragment_type) {
				access_fragment_t *fragment = (access_fragment_t *) newFragment;
				assert(fragment->_taskGroup != nullptr);
				
				task_group_t *taskGroup = fragment->_taskGroup;
				assert(!taskGroup->_liveFragments.contains(_newRegion));
				taskGroup->_liveFragments.insert(AccessFragmentWrapper(fragment));
			} else if ((newFragment->_objectType == taskwait_type) || (newFragment->_objectType == top_level_sink_type)) {
				taskwait_fragment_t *taskwaitFragment = (taskwait_fragment_t *) newFragment;
				assert(taskwaitFragment->_taskGroup != nullptr);
				
				task_group_t *taskGroup = taskwaitFragment->_taskGroup;
				assert(!taskGroup->_liveTaskwaitFragments.contains(_newRegion));
				taskGroup->_liveTaskwaitFragments.insert(TaskwaitFragmentWrapper(taskwaitFragment));
			} else {
				assert(newFragment->_objectType == regular_access_type);
				task_info_t &taskInfo = _taskToInfoMap[newFragment->_originator];
				assert(!taskInfo._liveAccesses.contains(_newRegion));
				taskInfo._liveAccesses.insert(AccessWrapper(newFragment));
			}
			
			// Set up the links of the new fragment
			for (auto &nextIdAndLink : original->_nextLinks) {
				task_id_t nextTaskId = nextIdAndLink.first;
				link_to_next_t &link = nextIdAndLink.second;
				
				auto it = newFragment->_nextLinks.find(nextTaskId);
				if (it != newFragment->_nextLinks.end()) {
					link_to_next_t &targetLink = it->second;
					for (auto &predecessor : link._predecessors) {
						targetLink._predecessors.insert(predecessor);
					}
				} else {
					newFragment->_nextLinks.emplace(
						std::pair<task_id_t, link_to_next_t> (nextTaskId, link)
					); // A link not duplicated yet
				}
				newFragment->_nextLinks[nextTaskId] = link;
				
				if (link._status == created_link_status) {
					for (task_id_t predecessorId : link._predecessors) {
						task_info_t &predecessor = _taskToInfoMap[predecessorId];
						assert(predecessor._outputEdges[nextTaskId]._hasBeenMaterialized);
						if (original->weak()) {
							predecessor._outputEdges[nextTaskId]._activeWeakContributorLinks++;
						} else {
							predecessor._outputEdges[nextTaskId]._activeStrongContributorLinks++;
						}
					}
				}
			}
		}
		
		
		std::string fragment_data_access_step_t::describe()
		{
			access_t *newFragment = _accessIdToAccessMap[_newFragmentAccessId];
			assert(newFragment != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": fragments ";
			
			switch (newFragment->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _originalAccessId << " of task " << newFragment->_originator << " into " << _newFragmentAccessId << " with region [" << _newRegion << "]";
			return oss.str();
		}
		
		
		bool fragment_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void create_subaccess_fragment_step_t::execute()
		{
			access_t *original = _accessIdToAccessMap[_accessId];
			access_fragment_t *newSubaccessFragment = (access_fragment_t *) _accessIdToAccessMap[_fragmentAccessId];
			assert(original != nullptr);
			assert(newSubaccessFragment != nullptr);
			
			newSubaccessFragment->_type = original->_type;
			newSubaccessFragment->_accessRegion = original->_accessRegion;
			newSubaccessFragment->weak() = original->weak();
			assert(newSubaccessFragment->_objectType == entry_fragment_type);
			newSubaccessFragment->satisfied() = original->satisfied();
			newSubaccessFragment->_status = created_access_status;
			
			assert(newSubaccessFragment->_taskGroup != nullptr);
			task_group_t *taskGroup = newSubaccessFragment->_taskGroup;
			assert(!taskGroup->_liveFragments.contains(newSubaccessFragment->_accessRegion));
			taskGroup->_liveFragments.insert(AccessFragmentWrapper(newSubaccessFragment));
		}
		
		
		std::string create_subaccess_fragment_step_t::describe()
		{
			access_t *original = _accessIdToAccessMap[_accessId];
			assert(original != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": creates entry fragment " << _fragmentAccessId << " from access " << _accessId << " of task " << original->_originator;
			return oss.str();
		}
		
		
		bool create_subaccess_fragment_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void completed_data_access_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->complete() = true;
		}
		
		
		std::string completed_data_access_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": completes ";
			
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId;
			
			return oss.str();
		}
		
		
		bool completed_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void data_access_becomes_removable_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_status = removable_access_status;
		}
		
		
		std::string data_access_becomes_removable_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": makes ";
			
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId << " of task " << access->_originator << " become removable";
			return oss.str();
		}
		
		
		bool data_access_becomes_removable_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void removed_data_access_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_status = removed_access_status;
			// 			for (auto &nextAccessLink : access->_nextLinks) {
			// 				nextAccessLink.second._status = dead_link_status;
			// 			}
		}
		
		
		std::string removed_data_access_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": removes ";
			
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId << " of task " << access->_originator;
			
			return oss.str();
		}
		
		
		bool removed_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void linked_data_accesses_step_t::execute()
		{
			access_t *sourceAccess = _accessIdToAccessMap[_sourceAccessId];
			assert(sourceAccess != nullptr);
			
			link_to_next_t &link = sourceAccess->_nextLinks[_sinkTaskId];
			link._status = created_link_status;
			
			if ((link._sinkObjectType == taskwait_type) || (link._sinkObjectType == top_level_sink_type)) {
				return;
			}
			
			for (task_id_t predecessorId : link._predecessors) {
				task_info_t &predecessor = _taskToInfoMap[predecessorId];
				predecessor._outputEdges[_sinkTaskId]._hasBeenMaterialized = true;
				if (sourceAccess->weak()) {
					_producedChanges |= (predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks == 0) && (predecessor._outputEdges[_sinkTaskId]._activeWeakContributorLinks == 0);
					predecessor._outputEdges[_sinkTaskId]._activeWeakContributorLinks++;
				} else {
					_producedChanges |= (predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks == 0);
					predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks++;
				}
			}
		}
		
		
		std::string linked_data_accesses_step_t::describe()
		{
			access_t *sourceAccess = _accessIdToAccessMap[_sourceAccessId];
			assert(sourceAccess != nullptr);
			link_to_next_t &link = sourceAccess->_nextLinks[_sinkTaskId];
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": links ";
			
			switch (sourceAccess->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _sourceAccessId << " of task " << sourceAccess->_originator;
			
			if ((link._sinkObjectType == regular_access_type) || (link._sinkObjectType == entry_fragment_type)) {
				oss << " to task " << _sinkTaskId;
			} else {
				oss << " to taskwait from task " << _sinkTaskId;
			}
			
			oss << " over region [" << _region << "]";
			return oss.str();
		}
		
		
		bool linked_data_accesses_step_t::visible()
		{
			return _showDependencyStructures || _producedChanges;
		}
		
		
		
		void unlinked_data_accesses_step_t::execute()
		{
			access_t *sourceAccess = _accessIdToAccessMap[_sourceAccessId];
			assert(sourceAccess != nullptr);
			
			link_to_next_t &link = sourceAccess->_nextLinks[_sinkTaskId];
			link._status = dead_link_status;
			
			if ((link._sinkObjectType == taskwait_type) || (link._sinkObjectType == top_level_sink_type)) {
				return;
			}
			
			for (task_id_t predecessorId : link._predecessors) {
				task_info_t &predecessor = _taskToInfoMap[predecessorId];
				if (sourceAccess->weak()) {
					predecessor._outputEdges[_sinkTaskId]._activeWeakContributorLinks--;
					_producedChanges |= (predecessor._outputEdges[_sinkTaskId]._activeWeakContributorLinks == 0) && (predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks == 0);
				} else {
					predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks--;
					_producedChanges |= (predecessor._outputEdges[_sinkTaskId]._activeStrongContributorLinks == 0);
				}
			}
		}
		
		
		std::string unlinked_data_accesses_step_t::describe()
		{
			access_t *sourceAccess = _accessIdToAccessMap[_sourceAccessId];
			assert(sourceAccess != nullptr);
			link_to_next_t &link = sourceAccess->_nextLinks[_sinkTaskId];
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": unlinks ";
			
			switch (sourceAccess->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _sourceAccessId << " of task " << sourceAccess->_originator;
			
			if ((link._sinkObjectType == regular_access_type) || (link._sinkObjectType == entry_fragment_type)) {
				oss << " from task " << _sinkTaskId;
			} else {
				oss << " from taskwait in " << _sinkTaskId;
			}
			return oss.str();
		}
		
		
		bool unlinked_data_accesses_step_t::visible()
		{
			return _showDependencyStructures || _producedChanges;
		}
		
		
		
		void reparented_data_access_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			assert(access->_superAccess == _oldSuperAccessId);
			access->_superAccess = _newSuperAccessId;
		}
		
		
		std::string reparented_data_access_step_t::describe()
		{
			access_t *original = _accessIdToAccessMap[_accessId];
			access_t *oldSuperAccess = _accessIdToAccessMap[_oldSuperAccessId];
			assert(original != nullptr);
			assert(oldSuperAccess != nullptr);
			
			access_t *newSuperAccess = nullptr;
			if (_newSuperAccessId != data_access_id_t()) {
				newSuperAccess = _accessIdToAccessMap[_newSuperAccessId];
				assert(newSuperAccess != nullptr);
			}
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": moves access " << _accessId << " of task " << original->_originator << " from son of task " << oldSuperAccess->_originator << " to ";
			if (newSuperAccess != nullptr) {
				oss << newSuperAccess->_originator;
			} else {
				oss << "top level";
			}
			
			return oss.str();
		}
		
		
		bool reparented_data_access_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void new_data_access_property_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_otherProperties.insert(_shortName);
		}
		
		
		std::string new_data_access_property_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			emitCPUAndTask(oss);
			oss << ": ";
			
			switch (access->_objectType) {
				case regular_access_type:
					oss << "access";
					break;
				case entry_fragment_type:
					oss << "entry fragment";
					break;
				case taskwait_type:
					oss << "taskwait";
					break;
				case top_level_sink_type:
					oss << "top level sink";
					break;
			}
			
			oss << " " << _accessId << " of task " << access->_originator << " has new property [" << _longName << " (" << _shortName <<") ]";
			return oss.str();
		}
		
		
		bool new_data_access_property_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void log_message_step_t::execute()
		{
		}
		
		
		std::string log_message_step_t::describe()
		{
			std::ostringstream oss;
			
			if (_instrumentationContext._externalThreadName == nullptr) {
				oss << "CPU " << _instrumentationContext._computePlaceId;
				if (_instrumentationContext._taskId != task_id_t()) {
					oss << " task " << _instrumentationContext._taskId;
				}
			} else {
				oss << "ExternalThread " << *_instrumentationContext._externalThreadName;
			}
			
			oss << ": " << _message;
			
			return oss.str();
		}
		
		
		bool log_message_step_t::visible()
		{
			return _showLog;
		}
		
	}
}
