#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP


#include <cassert>
#include <deque>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentTaskId.hpp>


class DataAccessRegistration {
private:
	static inline void reevaluateAndPropagateSatisfiability(Instrument::task_id_t instrumentationTaskId, DataAccess *previousDataAccess, DataAccess *targetDataAccess, WorkerThread::satisfied_originator_list_t /* OUT */ &satisfiedOriginators)
	{
		if (targetDataAccess == nullptr) {
			return;
		}
		auto position = targetDataAccess->_dataAccessSequence->_accessSequence.iterator_to(*targetDataAccess);
		
		while (position != targetDataAccess->_dataAccessSequence->_accessSequence.end()) {
			DataAccess *currentDataAccess = &(*position);
			assert(currentDataAccess != nullptr);
			
			bool becomesSatisfied = currentDataAccess->reevaluateSatisfiability(previousDataAccess);
			if (becomesSatisfied) {
				Instrument::dataAccessBecomesSatisfied(
					currentDataAccess->_instrumentationId,
					false, false, true,
					instrumentationTaskId,
					currentDataAccess->_originator->getInstrumentationTaskId()
				);
				
				if (currentDataAccess->_weak) {
					DataAccessSequence &subaccesses = currentDataAccess->_subaccesses;
					auto it = subaccesses._accessSequence.begin();
					if (it != subaccesses._accessSequence.end()) {
						reevaluateAndPropagateSatisfiability(instrumentationTaskId, previousDataAccess, &(*it), satisfiedOriginators);
					}
				} else {
					satisfiedOriginators.push_back(currentDataAccess->_originator);
				}
			} else {
				// Either it was already satisfied or it cannot become satisfied
				break;
			}
			
			previousDataAccess = currentDataAccess;
			position++;
		}
	}
	
	
	static inline void unregisterDataAccess(Instrument::task_id_t instrumentationTaskId, DataAccess *dataAccess, WorkerThread::satisfied_originator_list_t /* OUT */ &satisfiedOriginators)
	{
		assert(dataAccess != nullptr);
		
		DataAccessSequence &subaccesses = dataAccess->_subaccesses;
		// Locking strategy:
		// 	Every DataAccess that accesses the same data is protected by the same SpinLock that is located
		// 	in the DataAccess map together with the root DataAccessSequence of the data. Each DataAccessSequence,
		// 	a pointer to the SpinLock. However, since a DataAccess can be moved from one sequence to another
		// 	we cannot rely on getting the root spinlock from the sequence of the DataAccess since the sequence
		// 	may disappear while we attempt to grab the lock. Instead we get it from the subaccesses, which
		// 	is actually an embedded DataAccessSequence and should have the correct pointer.
		{
			std::unique_lock<SpinLock> guard(subaccesses.getLockGuard());
			
			DataAccessSequence *dataAccessSequence = dataAccess->_dataAccessSequence;
			
			DataAccessSequence::access_sequence_t::iterator dataAccessPosition = dataAccessSequence->_accessSequence.iterator_to(*dataAccess);
			
			auto nextPosition = dataAccessSequence->_accessSequence.erase(dataAccessPosition);
			
			// Move the subaccesses to the location where the DataAccess was
			auto newPreviousPosition = dataAccessSequence->_accessSequence.end();
			auto current = subaccesses._accessSequence.begin();
			while (current != subaccesses._accessSequence.end()) {
				DataAccess &subaccess = *current;
				
				Instrument::reparentedDataAccess(
					dataAccess->_instrumentationId,
					(dataAccessSequence->_superAccess != nullptr ? dataAccessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
					subaccess._instrumentationId,
					instrumentationTaskId
				);
				
				subaccess._dataAccessSequence = dataAccessSequence;
				
				if ((current == subaccesses._accessSequence.begin()) && (nextPosition != dataAccessSequence->_accessSequence.begin())) {
					auto previousOfFirstPosition = nextPosition;
					previousOfFirstPosition--;
					
					DataAccess *previousOfFirst = &(*previousOfFirstPosition);
					assert(previousOfFirst != nullptr);
					Instrument::linkedDataAccesses(
						previousOfFirst->_instrumentationId,
						subaccess._instrumentationId,
						dataAccessSequence->_accessRange,
						true, true,
						instrumentationTaskId
					);
				}
				
				current = subaccesses._accessSequence.erase(current);
				newPreviousPosition = dataAccessSequence->_accessSequence.insert(nextPosition, subaccess);
				
				if (nextPosition != dataAccessSequence->_accessSequence.end()) {
					auto positionOfNextToCurrent = current;
					positionOfNextToCurrent++;
					
					DataAccess *next = &(*nextPosition);
					assert(next != nullptr);
					
					if (positionOfNextToCurrent == subaccesses._accessSequence.end()) {
						// The current subaccess is the last one
						Instrument::linkedDataAccesses(
							subaccess._instrumentationId,
							next->_instrumentationId,
							dataAccessSequence->_accessRange,
							true, true,
							instrumentationTaskId
						);
					}
				}
			}
			
			// Instrumenters first see the movement, then the deletion
			Instrument::removedDataAccess(
				dataAccess->_instrumentationId,
				instrumentationTaskId
			);
			
			if ((nextPosition != dataAccessSequence->_accessSequence.end()) && !nextPosition->_satisfied) {
				// There is a next element in the sequence that must be reevaluated
				DataAccess *next = &(*nextPosition);
				assert(next != nullptr);
				
				DataAccess *effectivePrevious = nullptr;
				if (newPreviousPosition != dataAccessSequence->_accessSequence.end()) {
					effectivePrevious = &(*newPreviousPosition);
					assert(effectivePrevious != nullptr);
				} else {
					effectivePrevious = dataAccessSequence->getEffectivePrevious(next);
					if (effectivePrevious != nullptr) {
						Instrument::linkedDataAccesses(
							effectivePrevious->_instrumentationId,
							next->_instrumentationId,
							dataAccessSequence->_accessRange,
							false /* not direct */,
							false /* unidirectional */,
							instrumentationTaskId
						);
					}
				}
				
				reevaluateAndPropagateSatisfiability(instrumentationTaskId, effectivePrevious, next, satisfiedOriginators);
			}
		}
	}
	
	
	//! Process all the originators for whose a DataAccess has become satisfied
	static inline void processSatisfiedOriginators(WorkerThread::satisfied_originator_list_t &satisfiedOriginators, HardwarePlace *hardwarePlace)
	{
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			bool becomesReady = satisfiedOriginator->decreasePredecessors();
			if (becomesReady) {
				HardwarePlace *idleHardwarePlace = Scheduler::addReadyTask(satisfiedOriginator, hardwarePlace);
				
				if (idleHardwarePlace != nullptr) {
					ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
				}
			}
		}
	}
	
	
public:

	//! \brief adds a task access to the end of a sequence taking into account repeated accesses
	//! 
	//! \param[in] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] weak true iff the access is weak
	//! \param[in] accessSequence the sequnece on which to add the new access
	//! \param[out] dataAccess gets initialized with a pointer to the new DataAccess object or null if there was already a previous one for that task
	//! 
	//! \returns true is the access can be started
	//!
	//! The new DataAccess object has the task as its originator and is inserted in the DataAccessSequence.
	//! However, it is not inserted in the list of accesses of the Task.
	//! 
	//! If the task has already a previous access, it may be upgraded if necessary, and dataAccess is set to null. The return
	//! value indicates if the new access produces an additional dependency (only possible if the previous one did not).
	static inline bool registerTaskDataAccess(Task *task, DataAccessType accessType, bool weak, DataAccessSequence *accessSequence, DataAccess *&dataAccess)
	{
		assert(task != 0);
		assert(accessSequence != nullptr);
		
		std::unique_lock<SpinLock> guard(accessSequence->getLockGuard());
		
		DataAccess *effectivePrevious;
		if (!accessSequence->_accessSequence.empty()) {
			auto lastPosition = accessSequence->_accessSequence.rbegin();
			effectivePrevious = &(*lastPosition);
			assert(effectivePrevious != nullptr);
			
			if (effectivePrevious->_originator == task) {
				// The task "accesses" twice to the same location
				
				dataAccess = 0;
				return DataAccess::upgradeAccess(task, effectivePrevious, accessType, weak);
			}
		} else {
			// New access to an empty sequence
			effectivePrevious = accessSequence->getEffectivePrevious(nullptr);
		}
		
		bool satisfied;
		if (effectivePrevious != nullptr) {
			satisfied = DataAccess::evaluateSatisfiability(effectivePrevious, accessType);
		} else {
			// We no longer have (or never had) information about any previous access to this storage
			satisfied = true;
			Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), accessSequence, true);
		}
		
		Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
			(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
			accessType, weak,
			accessSequence->_accessRange,
			false, false, satisfied,
			task->getInstrumentationTaskId()
		);
		if (effectivePrevious != nullptr) {
			Instrument::linkedDataAccesses(
				effectivePrevious->_instrumentationId,
				dataAccessInstrumentationId,
				accessSequence->_accessRange,
				!accessSequence->_accessSequence.empty() /* Direct? */,
				!accessSequence->_accessSequence.empty() /* Bidirectional? */,
				task->getInstrumentationTaskId()
			);
		}
		Instrument::addTaskToAccessGroup(accessSequence, task->getInstrumentationTaskId());
		
		dataAccess = new DataAccess(accessSequence, accessType, weak, satisfied, task, accessSequence->_accessRange, dataAccessInstrumentationId);
		accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
		
		return satisfied || weak;
	}
	
	
	//! \brief Performs the task dependency registration procedure
	//! 
	//! \param[in] task the Task whose dependencies need to be calculated
	//! 
	//! \returns true if the task is already ready
	static inline bool registerTaskDataAccesses(Task *task)
	{
		assert(task != 0);
		
		// We increase the number of predecessors to avoid having the task become ready while we are adding its dependencies.
		// We do it by 2 because we add the data access and unlock access to it before increasing the number of predecessors.
		task->increasePredecessors(2);
		
		nanos_task_info *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		taskInfo->register_depinfo(task, task->getArgsBlock());
		
		return task->decreasePredecessors(2);
	}
	
	
	static inline void unregisterTaskDataAccesses(Task *finishedTask)
	{
		assert(finishedTask != 0);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != 0);
		HardwarePlace *hardwarePlace = currentThread->getHardwarePlace();
		assert(hardwarePlace != 0);
		
		TaskDataAccesses &taskDataAccesses = finishedTask->getDataAccesses();
		
		// A temporary list of tasks to minimize the time spent with the mutex held.
		WorkerThread::satisfied_originator_list_t &satisfiedOriginators = currentThread->getSatisfiedOriginatorsReference();
		for (auto it = taskDataAccesses.begin(); it != taskDataAccesses.end(); it = taskDataAccesses.erase(it)) {
			DataAccess *dataAccess = &(*it);
			
			assert(dataAccess->_originator == finishedTask);
			unregisterDataAccess(finishedTask->getInstrumentationTaskId(), dataAccess, /* OUT */ satisfiedOriginators);
			
			processSatisfiedOriginators(satisfiedOriginators, hardwarePlace);
			satisfiedOriginators.clear();
			
			// FIXME: delete the access
		}
	}
};


#endif // DATA_ACCESS_REGISTRATION_HPP

