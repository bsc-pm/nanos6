#include "InstrumentTaskId.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentDependenciesByAccess.hpp"

#include "dependencies/DataAccessType.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>
#include <mutex>



namespace Instrument {
	using namespace Graph;
	
	
	void registerTaskAccessInDependencyInfo(dependency_info_t &dependencyInfo, task_group_t *taskGroup, task_id_t taskId, DataAccessType accessType)
	{
		if (accessType == WRITE_ACCESS_TYPE) {
			// Handle repeated accesses
			if (dependencyInfo._lastWriter == taskId) {
				return;
			}
			
			// Handle access upgrades
			dependencyInfo._lastReaders.erase(taskId);
			
			if (dependencyInfo._lastAccessType == WRITE_ACCESS_TYPE) {
				if (dependencyInfo._lastWriter != -1) {
					taskGroup->_dependenciesGroupedBySource[dependencyInfo._lastWriter].insert(taskId);
				}
			} else {
				assert(dependencyInfo._lastAccessType == READ_ACCESS_TYPE);
				for (auto predecessor : dependencyInfo._lastReaders) {
					taskGroup->_dependenciesGroupedBySource[predecessor].insert(taskId);
				}
			}
			
			dependencyInfo._lastWriter = taskId;
			dependencyInfo._lastAccessType = WRITE_ACCESS_TYPE;
			dependencyInfo._lastReaders.clear();
		} else {
			assert(accessType == READ_ACCESS_TYPE);
			
			// Handle access upgrades
			if (dependencyInfo._lastWriter == taskId) {
				return;
			}
			
			// Handle repeated accesses
			if (dependencyInfo._lastReaders.find(taskId) != dependencyInfo._lastReaders.end()) {
				return;
			}
			
			if (dependencyInfo._lastWriter != -1) {
				taskGroup->_dependenciesGroupedBySource[dependencyInfo._lastWriter].insert(taskId);
			}
			
			assert((dependencyInfo._lastAccessType == READ_ACCESS_TYPE) || dependencyInfo._lastReaders.empty());
			dependencyInfo._lastAccessType = READ_ACCESS_TYPE;
			dependencyInfo._lastReaders.insert(taskId);
		}
	}
	
	
	void registerTaskAccess(task_id_t taskId, DataAccessType accessType, __attribute__((unused)) bool weak, void *start, size_t length)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			assert(_nextTaskId == 0);
		}
		
		task_id_t parentTaskId = -1;
		if (currentThread != nullptr) {
			Task *parentTask = currentThread->getTask();
			if (parentTask != nullptr) {
				parentTaskId = parentTask->getInstrumentationTaskId();
			} else {
				return;
			}
		} else {
			return;
		}
		assert(parentTaskId != -1);
		
		task_info_t &parentInfo = _taskToInfoMap[parentTaskId];
		
		// FIXME: When we delay pulling out dependency information the following code will break since it assumes the tasks are in the last phase
		task_group_t *taskGroup = nullptr;
		if (!parentInfo._phaseList.empty()) {
			phase_t *lastPhase = parentInfo._phaseList.back();
			taskGroup = dynamic_cast<task_group_t *>(lastPhase);
		}
		assert(taskGroup != nullptr);
		
		if (accessType == READWRITE_ACCESS_TYPE) {
			accessType = WRITE_ACCESS_TYPE;
		}
		
		DataAccessRange accessRange(start, length);
		taskGroup->_dependencyInfoMap.processIntersectingAndMissing(
			accessRange,
			// Information that overlaps in the current map
			[&](dependency_info_map_t::iterator position) -> bool {
				// Fragment the node as needed
				dependency_info_map_t::iterator interestingFragmentPosition
					= taskGroup->_dependencyInfoMap.fragmentByIntersection(position, accessRange, false);
					assert(interestingFragmentPosition != taskGroup->_dependencyInfoMap.end());
				
				dependency_info_t &dependencyInfo = *interestingFragmentPosition;
				registerTaskAccessInDependencyInfo(
					dependencyInfo, taskGroup,
					taskId, accessType
				);
				
				return true;
			},
			// Range not currently in the map
			[&](DataAccessRange missingRange) -> bool {
				// Add the range to the map
				dependency_info_map_t::iterator newDependencyInfoPosition
					= taskGroup->_dependencyInfoMap.insert( dependency_info_t(missingRange) );
				assert(newDependencyInfoPosition != taskGroup->_dependencyInfoMap.end());
				
				dependency_info_t &dependencyInfo = *newDependencyInfoPosition;
				registerTaskAccessInDependencyInfo(
					dependencyInfo, taskGroup,
					taskId, accessType
				);
				
				return true;
			}
		);
	}
}

