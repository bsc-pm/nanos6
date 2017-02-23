#include <cassert>
#include <sstream>
#include <string>

#include "ExecutionSteps.hpp"
#include "InstrumentGraph.hpp"


namespace Instrument {
	namespace Graph {
		
		
		void create_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == not_created_status);
			
			taskInfo._status = not_started_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string create_task_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << ": start creating task " << _triggererTaskId;
			return oss.str();
		}
		
		
		bool create_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void enter_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert((taskInfo._status == not_started_status) || (taskInfo._status == blocked_status));
			
			taskInfo._status = started_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string enter_task_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << ": enter task " << _triggererTaskId;
			return oss.str();
		}
		
		
		bool enter_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void exit_task_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = finished_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string exit_task_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << ": exit task " << _triggererTaskId;
			return oss.str();
		}
		
		
		bool exit_task_step_t::visible()
		{
			return true;
		}
		
		
		
		void enter_taskwait_step_t::execute()
		{
			if (_triggererTaskId == task_id_t()) {
				return;
			}
			
			taskwait_status_t &taskwaitStatus = _taskwaitStatus[_taskwaitId];
			taskwaitStatus._status = started_status;
			taskwaitStatus._lastCPU = _cpu;
			
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = blocked_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string enter_taskwait_step_t::describe()
		{
			if (_triggererTaskId == task_id_t()) {
				return "An external thread enters a taskwait";
			}
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": enter taskwait " << _taskwaitId;
			return oss.str();
		}
		
		
		bool enter_taskwait_step_t::visible()
		{
			return (_triggererTaskId != task_id_t());
		}
		
		
		
		void exit_taskwait_step_t::execute()
		{
			if (_triggererTaskId == task_id_t()) {
				return;
			}
			
			taskwait_status_t &taskwaitStatus = _taskwaitStatus[_taskwaitId];
			taskwaitStatus._status = finished_status;
			taskwaitStatus._lastCPU =_cpu;
			
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == blocked_status);
			taskInfo._status = started_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string exit_taskwait_step_t::describe()
		{
			if (_triggererTaskId == task_id_t()) {
				return "An external thread exits a taskwait";
			}
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": exit taskwait " << _taskwaitId;
			return oss.str();
		}
		
		
		bool exit_taskwait_step_t::visible()
		{
			return (_triggererTaskId != task_id_t());
		}
		
		
		
		void enter_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert((taskInfo._status == started_status) || (taskInfo._status == blocked_status));
			
			taskInfo._status = started_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string enter_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": enter usermutex " << _usermutexId;
			return oss.str();
		}
		
		
		bool enter_usermutex_step_t::visible()
		{
			return true;
		}
		
		
		
		void block_on_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == started_status);
			
			taskInfo._status = blocked_status;
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string block_on_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": blocks on usermutex " << _usermutexId;
			return oss.str();
		}
		
		
		bool block_on_usermutex_step_t::visible()
		{
			return true;
		}
		
		
		
		void exit_usermutex_step_t::execute()
		{
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(taskInfo._status == started_status);
			
			if (taskInfo._lastCPU == _cpu) {
				// Not doing anything for now
				// Perhaps will represent the state of the mutex, its allocation slots,
				// and links from those to task-internal critical nodes
			}
			taskInfo._lastCPU = _cpu;
		}
		
		
		std::string exit_usermutex_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": exit usermutex " << _usermutexId;
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
			access->_originator = _triggererTaskId;
			access->_type = _accessType;
			access->_accessRange = _range;
			access->weak() = _weak;
			access->readSatisfied() = _readSatisfied;
			access->writeSatisfied() = _writeSatisfied;
			access->satisfied() = _globallySatisfied;
			access->_status = created_access_status;
			
			task_info_t &taskInfo = _taskToInfoMap[_triggererTaskId];
			assert(!taskInfo._liveAccesses.contains(_range));
			taskInfo._liveAccesses.insert(AccessWrapper(access));
		}
		
		
		std::string create_data_access_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": creates";
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
			}
			oss << " access " << _accessId;
			oss << " [" << _range << "]";
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": upgrades access " << _accessId << " to";
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
			
			access->readSatisfied() = access->readSatisfied() | _readSatisfied;
			access->writeSatisfied() = access->writeSatisfied() | _writeSatisfied;
			access->satisfied() = access->satisfied() | _globallySatisfied;
		}
		
		
		std::string data_access_becomes_satisfied_step_t::describe()
		{
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": makes access " << _accessId << " of task " << _targetTaskId;
			if (_readSatisfied) {
				oss << " read";
			}
			if (_writeSatisfied) {
				oss << " write";
			}
			if (_globallySatisfied) {
				oss << " globally";
			}
			oss << " satisfied";
			return oss.str();
		}
		
		
		bool data_access_becomes_satisfied_step_t::visible()
		{
			return _showDependencyStructures;
		}
		
		
		
		void modified_data_access_range_step_t::execute()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			access->_accessRange = _range;
			
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
		
		
		std::string modified_data_access_range_step_t::describe()
		{
			access_t *access = _accessIdToAccessMap[_accessId];
			assert(access != nullptr);
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": changes range of "
				<< (access->fragment() ? "entry fragment " : "access ") << _accessId << " to [" << _range << "]";
			return oss.str();
		}
		
		
		bool modified_data_access_range_step_t::visible()
		{
			return _showDependencyStructures && _showRanges;
		}
		
		
		
		void fragment_data_access_step_t::execute()
		{
			access_t *original = _accessIdToAccessMap[_originalAccessId];
			access_t *newFragment = _accessIdToAccessMap[_newFragmentAccessId];
			assert(original != nullptr);
			assert(newFragment != nullptr);
			
			newFragment->_type = original->_type;
			newFragment->_accessRange = _newRange;
			newFragment->_bitset = original->_bitset;
			newFragment->_status = original->_status;
			
			if (newFragment->fragment()) {
				access_fragment_t *fragment = (access_fragment_t *) newFragment;
				assert(fragment->_taskGroup != nullptr);
				
				task_group_t *taskGroup = fragment->_taskGroup;
				assert(!taskGroup->_liveFragments.contains(_newRange));
				taskGroup->_liveFragments.insert(AccessFragmentWrapper(fragment));
			} else {
				task_info_t &taskInfo = _taskToInfoMap[newFragment->_originator];
				assert(!taskInfo._liveAccesses.contains(_newRange));
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": fragments " << (newFragment->fragment() ? "entry fragment " : "access ") ;
			oss << _originalAccessId << " of task " << newFragment->_originator << " into " << _newFragmentAccessId << " with range [" << _newRange << "]";
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
			newSubaccessFragment->_accessRange = original->_accessRange;
			newSubaccessFragment->weak() = original->weak();
			assert(newSubaccessFragment->fragment());
			newSubaccessFragment->readSatisfied() = original->readSatisfied();
			newSubaccessFragment->writeSatisfied() = original->writeSatisfied();
			newSubaccessFragment->satisfied() = original->satisfied();
			newSubaccessFragment->_status = created_access_status;
			
			assert(newSubaccessFragment->_taskGroup != nullptr);
			task_group_t *taskGroup = newSubaccessFragment->_taskGroup;
			assert(!taskGroup->_liveFragments.contains(newSubaccessFragment->_accessRange));
			taskGroup->_liveFragments.insert(AccessFragmentWrapper(newSubaccessFragment));
		}
		
		
		std::string create_subaccess_fragment_step_t::describe()
		{
			access_t *original = _accessIdToAccessMap[_accessId];
			assert(original != nullptr);
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": creates entry fragment " << _fragmentAccessId << " from access " << _accessId << " of task " << original->_originator;
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": completes " << (access->fragment() ? "entry fragment " : "access ") << _accessId;
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": makes " << (access->fragment() ? "entry fragment " : "access ") << _accessId << " of task " << access->_originator << " become removable";
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": removes " << (access->fragment() ? "entry fragment " : "access ") << _accessId << " of task " << access->_originator;
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
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": links "
			<< (sourceAccess->fragment() ? "entry fragment " : "access ") << _sourceAccessId << " of task " << sourceAccess->_originator << " to task " << _sinkTaskId << " over range [" << _range << "]";
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
			
			std::ostringstream oss;
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": unlinks "
				<< (sourceAccess->fragment() ? "entry fragment " : "access ") << _sourceAccessId << " of task " << sourceAccess->_originator << " from task " << _sinkTaskId;
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
			
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": moves access " << _accessId << " of task " << original->_originator << " from son of task " << oldSuperAccess->_originator << " to ";
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
			oss << "CPU " << _cpu << " task " << _triggererTaskId << ": " << (access->fragment() ? "entry fragment " : "access ") << _accessId << " of task " << access->_originator << " has new property [" << _longName << " (" << _shortName <<") ]";
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
			oss << "CPU " << _cpu;
			if (_triggererTaskId != task_id_t()) {
				oss << " task " << _triggererTaskId;
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
