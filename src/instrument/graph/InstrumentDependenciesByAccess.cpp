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
	
	void registerTaskAccess(task_id_t taskId, DataAccessType accessType, void *start, __attribute__((unused)) size_t length)
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
		
		dependency_info_t &dependencyInfo = taskGroup->_dependencyInfoMap[start];
		if (accessType == WRITE_ACCESS_TYPE) {
			// Handle repeated accesses
			if (dependencyInfo._lastWriter == taskId) {
				return;
			}
			
			// Handle access upgrades
			dependencyInfo._lastReaders.erase(taskId);
			
			if (dependencyInfo._lastAccessType == WRITE_ACCESS_TYPE) {
				if (dependencyInfo._lastWriter != -1) {
					_taskToInfoMap[dependencyInfo._lastWriter]._hasSuccessors = true;
					_taskToInfoMap[taskId]._hasPredecessors = true;
					taskGroup->_dependencyEdges.push_back(edge_t(dependencyInfo._lastWriter, taskId));
				}
			} else {
				assert(dependencyInfo._lastAccessType == READ_ACCESS_TYPE);
				for (auto predecessor : dependencyInfo._lastReaders) {
					_taskToInfoMap[predecessor]._hasSuccessors = true;
					_taskToInfoMap[taskId]._hasPredecessors = true;
					taskGroup->_dependencyEdges.push_back(edge_t(predecessor, taskId));
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
				_taskToInfoMap[dependencyInfo._lastWriter]._hasSuccessors = true;
				_taskToInfoMap[taskId]._hasPredecessors = true;
				taskGroup->_dependencyEdges.push_back(edge_t(dependencyInfo._lastWriter, taskId));
			}
			
			assert((dependencyInfo._lastAccessType == READ_ACCESS_TYPE) || dependencyInfo._lastReaders.empty());
			dependencyInfo._lastAccessType = READ_ACCESS_TYPE;
			dependencyInfo._lastReaders.insert(taskId);
		}
	}
}

