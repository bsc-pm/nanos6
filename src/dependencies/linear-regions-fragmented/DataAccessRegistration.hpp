#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

#include <cassert>
#include <deque>
#include <mutex>
#include <vector>

#include "CPUDependencyData.hpp"
#include "DataAccess.hpp"

#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include "TaskDataAccessesImplementation.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentLogMessage.hpp>
#include <InstrumentTaskId.hpp>


class DataAccessRegistration {
public:
	typedef CPUDependencyData::removable_task_list_t removable_task_list_t;
	
	
private:
	typedef CPUDependencyData::DelayedOperation DelayedOperation;
	
	
	static inline DataAccess *createAccess(Task *originator, DataAccessType accessType, bool weak, DataAccessRange range, bool fragment)
	{
		Instrument::data_access_id_t newDataAccessInstrumentationId;
		
		// Regular object duplication
		DataAccess *dataAccess = new DataAccess(
			accessType, weak, originator, range,
			fragment,
			newDataAccessInstrumentationId
		);
		
		return dataAccess;
	}
	
	
	static inline void upgradeAccess(DataAccess *dataAccess, DataAccessType accessType, bool weak)
	{
		assert(dataAccess != nullptr);
		assert(!dataAccess->hasBeenDiscounted());
		
		bool newWeak = dataAccess->_weak && weak;
		
		DataAccessType newDataAccessType = accessType;
		if (accessType != dataAccess->_type) {
			newDataAccessType = READWRITE_ACCESS_TYPE;
		}
		
		if ((newWeak != dataAccess->_weak) || (newDataAccessType != dataAccess->_type)) {
			Instrument::upgradedDataAccess(
				dataAccess->_instrumentationId,
				dataAccess->_type, dataAccess->_weak,
				newDataAccessType, newWeak,
				false,
				dataAccess->_originator->getInstrumentationTaskId()
			);
			
			dataAccess->_type = newDataAccessType;
			dataAccess->_weak = newWeak;
		}
	}
	
	
	// NOTE: locking should be handled from the outside
	static inline DataAccess *duplicateDataAccess(
		DataAccess const &toBeDuplicated,
		TaskDataAccesses &accessStructures,
		bool updateTaskBlockingCount
	) {
		assert(toBeDuplicated._originator != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!toBeDuplicated.hasBeenDiscounted());
		
		// Regular object duplication
		DataAccess *newFragment = createAccess(
			toBeDuplicated._originator,
			toBeDuplicated._type, toBeDuplicated._weak, toBeDuplicated._range,
			toBeDuplicated.isFragment()
		);
		
		newFragment->_status = toBeDuplicated._status;
		newFragment->_next = toBeDuplicated._next;
		
		if (updateTaskBlockingCount && !toBeDuplicated.isFragment() && !newFragment->_weak && !newFragment->satisfied()) {
			toBeDuplicated._originator->increasePredecessors();
		}
		
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		if (!newFragment->isRemovable(newFragment->hasForcedRemoval())) {
			accessStructures._removalBlockers++;
		}
		
		return newFragment;
	}
	
	
#ifndef NDEBUG
	static bool noAccessIsReachable(TaskDataAccesses &accessStructures)
	{
		assert(!accessStructures.hasBeenDeleted());
		return accessStructures._accesses.processAll(
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				return !position->isReachable();
			}
		);
	}
#endif
	
	
	static inline DataAccess *fragmentAccess(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccess *dataAccess, DataAccessRange range,
		TaskDataAccesses &accessStructures,
		TaskDataAccesses &parentAccessStructures,
		bool considerTaskBlocking
	) {
		assert(dataAccess != nullptr);
		// assert(accessStructures._lock.isLockedByThisThread()); // Not necessary when fragmenting an access that is not reachable
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		assert(&dataAccess->_originator->getDataAccesses() == &accessStructures);
		assert(
			!dataAccess->isInBottomMap()
			|| dataAccess->isFragment()
			|| (&dataAccess->_originator->getParent()->getDataAccesses() == &parentAccessStructures)
		);
		assert(
			!dataAccess->isInBottomMap()
			|| !dataAccess->isFragment()
			|| (&dataAccess->_originator->getDataAccesses() == &parentAccessStructures)
		);
		assert(!accessStructures.hasBeenDeleted());
		assert(!dataAccess->hasBeenDiscounted());
		
		if (dataAccess->_range.fullyContainedIn(range)) {
			// Nothing to fragment
		} else if (dataAccess->isInBottomMap()) {
			assert(!parentAccessStructures.hasBeenDeleted());
			assert(parentAccessStructures._lock.isLockedByThisThread());
			
			TaskDataAccesses::subaccess_bottom_map_t::iterator position =
				parentAccessStructures._subaccessBottomMap.iterator_to(*dataAccess);
			position = parentAccessStructures._subaccessBottomMap.fragmentByIntersection(
				position, range,
				false,
				[&](DataAccess const &toBeDuplicated) -> DataAccess * {
					return duplicateDataAccess(toBeDuplicated, accessStructures, considerTaskBlocking);
				},
				[&](DataAccess *fragment, DataAccess *originalDataAccess) {
					// Insert it back into the task accesses or access fragments
					assert(fragment != nullptr);
					assert(fragment->_originator != nullptr);
					TaskDataAccesses &originatorAccessStructures = fragment->_originator->getDataAccesses();
					assert(!originatorAccessStructures.hasBeenDeleted());
					
					if (fragment == originalDataAccess) {
						// Remove the access/fragment since it may have shifted its position. Will re-add later
						if (fragment->isFragment()) {
							originatorAccessStructures._accessFragments.erase(*fragment);
						} else {
							originatorAccessStructures._accesses.erase(*fragment);
						}
						
						Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
					} else {
						fragment->_instrumentationId =
							Instrument::fragmentedDataAccess(originalDataAccess->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
					}
					
					if (fragment->isFragment()) {
						originatorAccessStructures._accessFragments.insert(*fragment);
					} else {
						originatorAccessStructures._accesses.insert(*fragment);
					}
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->_range.fullyContainedIn(range));
		} else {
			if (dataAccess->isFragment()) {
				TaskDataAccesses::access_fragments_t::iterator position =
					accessStructures._accessFragments.iterator_to(*dataAccess);
				position = accessStructures._accessFragments.fragmentByIntersection(
					position, range,
					false,
					[&](DataAccess const &toBeDuplicated) -> DataAccess * {
						return duplicateDataAccess(toBeDuplicated, accessStructures, considerTaskBlocking);
					},
					[&](DataAccess *fragment, DataAccess *originalDataAccess) {
						if (fragment != originalDataAccess) {
							fragment->_instrumentationId =
								Instrument::fragmentedDataAccess(originalDataAccess->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
						} else {
							Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
						}
					}
				);
				
				dataAccess = &(*position);
				assert(dataAccess != nullptr);
				assert(dataAccess->_range.fullyContainedIn(range));
			} else {
				TaskDataAccesses::accesses_t::iterator position =
					accessStructures._accesses.iterator_to(*dataAccess);
				position = accessStructures._accesses.fragmentByIntersection(
					position, range,
					false,
					[&](DataAccess const &toBeDuplicated) -> DataAccess * {
						return duplicateDataAccess(toBeDuplicated, accessStructures, considerTaskBlocking);
					},
					[&](DataAccess *fragment, DataAccess *originalDataAccess) {
						if (fragment != originalDataAccess) {
							fragment->_instrumentationId =
								Instrument::fragmentedDataAccess(originalDataAccess->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
						} else {
							Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
						}
					}
				);
				
				dataAccess = &(*position);
				assert(dataAccess != nullptr);
				assert(dataAccess->_range.fullyContainedIn(range));
			}
		}
		
		return dataAccess;
	}
	
	
	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
		/* INOUT */ CPUDependencyData &cpuDependencyData,
		HardwarePlace *hardwarePlace
	) {
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : cpuDependencyData._satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			HardwarePlace *idleHardwarePlace = Scheduler::addReadyTask(satisfiedOriginator, hardwarePlace, SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT);
			
			if (idleHardwarePlace != nullptr) {
				ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
			}
		}
		
		cpuDependencyData._satisfiedOriginators.clear();
	}
	
	
	static inline DelayedOperation &getNewDelayedOperation(/* OUT */ CPUDependencyData &cpuDependencyData)
	{
		cpuDependencyData._delayedOperations.emplace_back();
		return cpuDependencyData._delayedOperations.back();
	}
	
	
	static inline void handleAccessRemoval(
		DataAccess *targetAccess, TaskDataAccesses &targetTaskAccessStructures, Task *targetTask,
		TaskDataAccesses *parentAccessStructures,
		Instrument::task_id_t triggererInstrumentationTaskId,
		bool allowRemovalFromBottomMap,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(targetTaskAccessStructures._removalBlockers > 0);
		targetTaskAccessStructures._removalBlockers--;
		targetAccess->markAsDiscounted();
		Instrument::dataAccessBecomesRemovable(targetAccess->_instrumentationId, triggererInstrumentationTaskId);
		
		if (targetAccess->_next != nullptr) {
			Instrument::unlinkedDataAccesses(
				targetAccess->_instrumentationId,
				targetAccess->_next->getInstrumentationTaskId(),
				/* direct */ true,
				triggererInstrumentationTaskId
			);
		} else {
			assert(targetAccess->isInBottomMap());
			if (allowRemovalFromBottomMap) {
				assert(parentAccessStructures != nullptr);
				targetAccess->isInBottomMap() = false;
				parentAccessStructures->_subaccessBottomMap.erase(targetAccess);
			}
		}
		
		if (targetTaskAccessStructures._removalBlockers == 0) {
			if (targetTask->decreaseRemovalBlockingCount()) {
				cpuDependencyData._removableTasks.push_back(targetTask);
			}
		}
		
		assert(targetAccess->hasForcedRemoval() || !targetAccess->isInBottomMap());
	}
	
	
	static inline void propagateSatisfiabilityToFragments(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(delayedOperation._target != nullptr);
		assert(delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]);
		
		Task *targetTask = delayedOperation._target;
		TaskDataAccesses &targetTaskAccessStructures = targetTask->getDataAccesses();
		assert(!targetTaskAccessStructures.hasBeenDeleted());
		assert(targetTaskAccessStructures._lock.isLockedByThisThread());
		
		// NOTE: An access is discounted before traversing the fragments, so by the time we reach this point, the counter could be 0
		
		targetTaskAccessStructures._accessFragments.processIntersecting(
			delayedOperation._range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *targetAccess = &(*position);
				assert(targetAccess != nullptr);
				assert(targetAccess->isFragment());
				assert(targetAccess->isReachable());
				assert(targetAccess->_originator == targetTask);
				assert(!targetAccess->hasBeenDiscounted());
				
				bool acceptsReadSatisfiability = !targetAccess->readSatisfied();
				bool acceptsWriteSatisfiability = !targetAccess->writeSatisfied();
				
				// Skip accesses whose state does not change
				if (
					(!delayedOperation._operation[DelayedOperation::PROPAGATE_READ] || !acceptsReadSatisfiability)
					&& (!delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] || !acceptsWriteSatisfiability)
				) {
					return true;
				}
				
				// Fragment if necessary
				if (!targetAccess->_range.fullyContainedIn(delayedOperation._range)) {
					targetAccess = fragmentAccess(
						triggererInstrumentationTaskId,
						targetAccess, delayedOperation._range,
						targetTaskAccessStructures,
						targetTaskAccessStructures /* This is used only for the bottom map */,
						/* Do not affect originator blocking counter */ false
					);
					assert(targetAccess != nullptr);
					assert(targetAccess->_range.fullyContainedIn(delayedOperation._range));
				}
				
				bool wasRemovable = targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				
				// Update the satisfiability
				if (delayedOperation._operation[DelayedOperation::PROPAGATE_READ] && acceptsReadSatisfiability) {
					assert(!targetAccess->readSatisfied());
					targetAccess->readSatisfied() = true;
				}
				if (delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] && acceptsWriteSatisfiability) {
					assert(!targetAccess->writeSatisfied());
					targetAccess->writeSatisfied() = true;
				}
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->_instrumentationId,
					delayedOperation._operation[DelayedOperation::PROPAGATE_READ] && acceptsReadSatisfiability,
					delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] && acceptsWriteSatisfiability,
					false,
					triggererInstrumentationTaskId,
					targetTask->getInstrumentationTaskId()
				);
				
				// Update the number of non removable accesses of the task
				if (!wasRemovable && targetAccess->isRemovable(targetAccess->hasForcedRemoval())) {
					handleAccessRemoval(
						targetAccess, targetTaskAccessStructures, targetTask,
						nullptr, triggererInstrumentationTaskId, /* Do not remove from bottom map */ false, cpuDependencyData
					);
				}
				
				assert((targetAccess->_next != nullptr) || targetAccess->isInBottomMap());
				
				// Propagates as is to subaccesses, and as a regular access to outer accesses
				bool propagationToSubaccess =
					(targetAccess->_next != nullptr) && (targetAccess->_next->getParent() == targetTask);
				
				bool canPropagateReadSatisfiability =
					targetAccess->readSatisfied()
					&& (targetAccess->complete() || (targetAccess->_type == READ_ACCESS_TYPE) || propagationToSubaccess);
				bool canPropagateWriteSatisfiability =
					targetAccess->writeSatisfied()
					&& (targetAccess->complete() || propagationToSubaccess);
				
				// Continue to next iteration if there is nothing to propagate
				if (
					(!canPropagateReadSatisfiability || !delayedOperation._operation[DelayedOperation::PROPAGATE_READ])
					&& (!canPropagateWriteSatisfiability || !delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE])
				) {
					return true;
				}
				
				if (targetAccess->_next != nullptr) {
					DelayedOperation &nextOperation = getNewDelayedOperation(cpuDependencyData);
					nextOperation._operation = delayedOperation._operation;
					nextOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS] = false;
					nextOperation._operation[DelayedOperation::PROPAGATE_READ] =
						nextOperation._operation[DelayedOperation::PROPAGATE_READ] & canPropagateReadSatisfiability;
					nextOperation._operation[DelayedOperation::PROPAGATE_WRITE] =
						nextOperation._operation[DelayedOperation::PROPAGATE_WRITE] & canPropagateWriteSatisfiability;
					nextOperation._range = targetAccess->_range;
					nextOperation._target = targetAccess->_next;
				}
				
				return true;
			}
		);
	}
	
	
	static inline void propagateSatisfiabilityPlain(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(delayedOperation._target != nullptr);
		assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]);
		
		Task *targetTask = delayedOperation._target;
		TaskDataAccesses &targetTaskAccessStructures = targetTask->getDataAccesses();
		TaskDataAccesses *parentAccessStructuresIfLocked = nullptr;
		assert(!targetTaskAccessStructures.hasBeenDeleted());
		assert(targetTaskAccessStructures._lock.isLockedByThisThread());
		
		targetTaskAccessStructures._accesses.processIntersectingWithRestart(
			delayedOperation._range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *targetAccess = &(*position);
				assert(targetAccess != nullptr);
				assert(targetAccess->isReachable());
				assert(targetAccess->_originator == targetTask);
				assert(!targetAccess->hasBeenDiscounted());
				
				bool acceptsReadSatisfiability = !targetAccess->readSatisfied();
				bool acceptsWriteSatisfiability = !targetAccess->writeSatisfied();
				
				// Skip accesses whose state does not change
				if (
					(!delayedOperation._operation[DelayedOperation::PROPAGATE_READ] || !acceptsReadSatisfiability)
					&& (!delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] || !acceptsWriteSatisfiability)
				) {
					return true;
				}
				
				// Fragment if necessary
				if (!targetAccess->_range.fullyContainedIn(delayedOperation._range) || targetAccess->hasForcedRemoval()) {
					if (targetAccess->isInBottomMap()) {
						if (parentAccessStructuresIfLocked == nullptr) {
							Task *targetParent = targetTask->getParent();
							assert(targetParent != nullptr);
							
							parentAccessStructuresIfLocked = &targetParent->getDataAccesses();
							assert(parentAccessStructuresIfLocked != nullptr);
							assert(!parentAccessStructuresIfLocked->hasBeenDeleted());
							
							if (!parentAccessStructuresIfLocked->_lock.tryLock()) {
								// Relock and restart
								targetTaskAccessStructures._lock.unlock();
								parentAccessStructuresIfLocked->_lock.lock();
								targetTaskAccessStructures._lock.lock();
								
								return false;
							}
						}
						assert(!parentAccessStructuresIfLocked->hasBeenDeleted());
					}
					
					targetAccess = fragmentAccess(
						triggererInstrumentationTaskId,
						targetAccess, delayedOperation._range, targetTaskAccessStructures,
						*parentAccessStructuresIfLocked,
						/* Affect originator blocking counter */ true
					);
					assert(targetAccess != nullptr);
				}
				
				bool wasSatisfied = targetAccess->satisfied();
				bool wasRemovable = targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				
				// Update the satisfiability
				if (delayedOperation._operation[DelayedOperation::PROPAGATE_READ] && acceptsReadSatisfiability) {
					assert(!targetAccess->readSatisfied());
					targetAccess->readSatisfied() = true;
				}
				if (delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] && acceptsWriteSatisfiability) {
					assert(!targetAccess->writeSatisfied());
					targetAccess->writeSatisfied() = true;
				}
				
				// Update the number of non removable accesses of the task
				if (!wasRemovable && targetAccess->isRemovable(targetAccess->hasForcedRemoval())) {
					handleAccessRemoval(
						targetAccess, targetTaskAccessStructures, targetTask,
						parentAccessStructuresIfLocked, triggererInstrumentationTaskId, true, cpuDependencyData
					);
				}
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->_instrumentationId,
					delayedOperation._operation[DelayedOperation::PROPAGATE_READ] && acceptsReadSatisfiability,
					delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] && acceptsWriteSatisfiability,
					false,
					triggererInstrumentationTaskId,
					targetTask->getInstrumentationTaskId()
				);
				
				
				// If the target access becomes satisfied decrement the predecessor count of the task
				// If it becomes 0 then add it to the list of satisfied originators
				if (!targetAccess->_weak && !wasSatisfied && targetAccess->satisfied()) {
					if (targetTask->decreasePredecessors()) {
						cpuDependencyData._satisfiedOriginators.push_back(targetTask);
					}
				}
				
				bool canPropagateReadSatisfiability =
					targetAccess->readSatisfied()
					&& (targetAccess->complete() || (targetAccess->_type == READ_ACCESS_TYPE));
				bool canPropagateWriteSatisfiability = targetAccess->writeSatisfied() && (targetAccess->complete());
				
				// Continue to next iteration if there is nothing to propagate
				if (
					(!canPropagateReadSatisfiability || !delayedOperation._operation[DelayedOperation::PROPAGATE_READ])
					&& (!canPropagateWriteSatisfiability || !delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE])
					&& !targetAccess->hasSubaccesses()
				) {
					return true;
				}
				
				// Propagate to fragments
				if (targetAccess->hasSubaccesses()) {
					DelayedOperation &operationForFragments = getNewDelayedOperation(cpuDependencyData);
					operationForFragments = delayedOperation;
					assert(targetAccess->_range.fullyContainedIn(delayedOperation._range));
					operationForFragments._range = targetAccess->_range;
					operationForFragments._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS] = true;
				} else if (targetAccess->_next != nullptr) {
					// Propagate to next
					DelayedOperation &operationForNext = getNewDelayedOperation(cpuDependencyData);
					operationForNext._operation = delayedOperation._operation;
					operationForNext._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS] = false;
					operationForNext._operation[DelayedOperation::PROPAGATE_READ] =
						operationForNext._operation[DelayedOperation::PROPAGATE_READ] & canPropagateReadSatisfiability;
					operationForNext._operation[DelayedOperation::PROPAGATE_WRITE] =
						operationForNext._operation[DelayedOperation::PROPAGATE_WRITE] & canPropagateWriteSatisfiability;
					operationForNext._range = targetAccess->_range;
					operationForNext._target = targetAccess->_next;
				} else {
					// For now let's handle deep propagation by deep linking on creation
					assert(targetAccess->_next == nullptr);
					// We cannot assert(targetAccess->isInBottomMap()), since during addition the new task is unlocked before adding its accesses to the bottom map
				}
				
				return true;
			}
		);
		
		// Unlock the parent (if locked)
		if (parentAccessStructuresIfLocked != nullptr) {
			parentAccessStructuresIfLocked->_lock.unlock();
		}
	}
	
	
	static inline void activateForcedRemovalOfBottomMapAccesses(
		Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range,
		/* OUT */ CPUDependencyData &cpuDependencyData,
		Instrument::task_id_t triggererInstrumentationTaskId
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		// For each bottom map access
		accessStructures._subaccessBottomMap.processIntersecting(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
				DataAccess *dataAccess = &(*position);
				assert(dataAccess != nullptr);
				assert(dataAccess->_next == nullptr);
				assert(dataAccess->isInBottomMap());
				assert(!dataAccess->hasBeenDiscounted());
				
				Task *subtask = dataAccess->_originator;
				assert(subtask != nullptr);
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				if (subtask != task) {
					subtaskAccessStructures._lock.lock();
				}
				
				assert(!dataAccess->hasForcedRemoval());
				
				dataAccess = fragmentAccess(
					triggererInstrumentationTaskId,
					dataAccess, range, subtaskAccessStructures,
					accessStructures,
					/* Affect originator blocking counter */ true
				);
				
				dataAccess->hasForcedRemoval() = true;
				
				if (!dataAccess->isFragment() && dataAccess->complete() && dataAccess->hasSubaccesses()) {
					activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, cpuDependencyData, triggererInstrumentationTaskId);
				}
				
				if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
					// The access has become removable
					handleAccessRemoval(
						dataAccess, subtaskAccessStructures, subtask,
						&accessStructures, triggererInstrumentationTaskId, true, cpuDependencyData
					);
				}
				
				if (subtask != task) {
					subtaskAccessStructures._lock.unlock();
				}
				
				return true;
			}
		);
	}
	
	
	static inline void activateForcedRemovalOfBottomMapAccesses(
		Task *task, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &cpuDependencyData,
		Instrument::task_id_t triggererInstrumentationTaskId
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		// For each bottom map access
		accessStructures._subaccessBottomMap.processAll(
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
				DataAccess *dataAccess = &(*position);
				assert(dataAccess != nullptr);
				assert(dataAccess->_next == nullptr);
				assert(dataAccess->isInBottomMap());
				assert(!dataAccess->hasBeenDiscounted());
				
				Task *subtask = dataAccess->_originator;
				assert(subtask != nullptr);
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				if (subtask != task) {
					subtaskAccessStructures._lock.lock();
				}
				
				assert(!dataAccess->hasForcedRemoval());
				dataAccess->hasForcedRemoval() = true;
				
				if (!dataAccess->isFragment() && dataAccess->complete() && dataAccess->hasSubaccesses()) {
					activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, cpuDependencyData, triggererInstrumentationTaskId);
				}
				
				if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
					// The access has become removable
					handleAccessRemoval(
						dataAccess, subtaskAccessStructures, subtask,
						&accessStructures, triggererInstrumentationTaskId, true, cpuDependencyData
					);
				}
				
				if (subtask != task) {
					subtaskAccessStructures._lock.unlock();
				}
				
				return true;
			}
		);
	}
	
	
	static void processDelayedOperation(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		if (delayedOperation._operation[DelayedOperation::LINK_BOTTOM_ACCESSES_TO_NEXT]) {
			assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]);
			
			processLinkingBottomMapAccessesToNextOperation(
				triggererInstrumentationTaskId,
				delayedOperation,
				cpuDependencyData
			);
		} else if (delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]) {
			propagateSatisfiabilityToFragments(
				triggererInstrumentationTaskId,
				delayedOperation,
				cpuDependencyData
			);
		} else {
			assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]);
			propagateSatisfiabilityPlain(
				triggererInstrumentationTaskId,
				delayedOperation,
				cpuDependencyData
			);
		}
	}
	
	
	static inline void processDelayedOperations(
		Instrument::task_id_t triggererInstrumentationTaskId,
		/* INOUT */ CPUDependencyData &cpuDependencyData
	) {
		Task *lastLocked = nullptr;
		
		while (!cpuDependencyData._delayedOperations.empty()) {
			DelayedOperation const &delayedOperation = cpuDependencyData._delayedOperations.front();
			
			assert(delayedOperation._target != nullptr);
			if (delayedOperation._target != lastLocked) {
				if (lastLocked != nullptr) {
					lastLocked->getDataAccesses()._lock.unlock();
				}
				lastLocked = delayedOperation._target;
				lastLocked->getDataAccesses()._lock.lock();
			}
			
			processDelayedOperation(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
			
			cpuDependencyData._delayedOperations.pop_front();
		}
		
		if (lastLocked != nullptr) {
			lastLocked->getDataAccesses()._lock.unlock();
		}
	}
	
	
	static void processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
		Instrument::task_id_t triggererInstrumentationTaskId,
		CPUDependencyData &cpuDependencyData,
		CPU *cpu,
		WorkerThread *currentThread
	) {
		processDelayedOperations(triggererInstrumentationTaskId, cpuDependencyData);
		processSatisfiedOriginators(cpuDependencyData, cpu);
		
		assert(cpuDependencyData._delayedOperations.empty());
		assert(cpuDependencyData._satisfiedOriginators.empty());
		
		handleRemovableTasks(cpuDependencyData._removableTasks, cpu, currentThread);
	}
	
	
	static inline DataAccess *createInitialFragment(
		Instrument::task_id_t triggererInstrumentationTaskId,
		TaskDataAccesses::accesses_t::iterator accessPosition,
		TaskDataAccesses &accessStructures
	) {
		DataAccess *dataAccess = &(*accessPosition);
		assert(dataAccess != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		
		assert(!accessStructures._accessFragments.contains(dataAccess->_range));
		
		Instrument::data_access_id_t instrumentationId =
			Instrument::createdDataSubaccessFragment(
				dataAccess->_instrumentationId,
				triggererInstrumentationTaskId
			);
		DataAccess *fragment = new DataAccess(
			dataAccess->_type,
			dataAccess->_weak,
			dataAccess->_originator,
			dataAccess->_range,
			/* A fragment */ true,
			instrumentationId
		);
		fragment->readSatisfied() = dataAccess->readSatisfied();
		fragment->writeSatisfied() = dataAccess->writeSatisfied();
		fragment->complete() = dataAccess->complete();
#ifndef NDEBUG
		fragment->isReachable() = true;
#endif
		
		accessStructures._accessFragments.insert(*fragment);
		accessStructures._subaccessBottomMap.insert(*fragment);
		fragment->isInBottomMap() = true;
		dataAccess->hasSubaccesses() = true;
		
		// Fragments also participate in the counter of non removable accesses
		accessStructures._removalBlockers++;
		
		return fragment;
	}
	
	
	template <typename MatchingProcessorType, typename MissingProcessorType>
	static inline bool foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRange(
		TaskDataAccesses &parentAccessStructures,
		Task *task, DataAccessRange range,
		MatchingProcessorType matchingProcessor, MissingProcessorType missingProcessor
	) {
		assert(task != nullptr);
		assert(!parentAccessStructures.hasBeenDeleted());
		
		return parentAccessStructures._subaccessBottomMap.processIntersectingAndMissing(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				DataAccess *previous = &(*bottomMapPosition);
				assert(previous != nullptr);
				
				return matchingProcessor(previous);
			},
			[&](DataAccessRange missingRange) -> bool {
				parentAccessStructures._accesses.processIntersectingAndMissing(
					missingRange,
					[&](TaskDataAccesses::accesses_t::iterator superaccessPosition) -> bool {
						DataAccess *previous = createInitialFragment(
							task->getInstrumentationTaskId(),
							superaccessPosition, parentAccessStructures
						);
						assert(previous != nullptr);
						
						return matchingProcessor(previous);
					},
					[&](DataAccessRange rangeUncoveredByParent) -> bool {
						return missingProcessor(rangeUncoveredByParent);
					}
				);
				
				return true;
			}
		);
	}
	
	
	static inline void createPropagationOperation(
		DataAccess *dataAccess,
		Task *targetTask,
		/* inout */ CPUDependencyData &cpuDependencyData
	) {
		assert(dataAccess != nullptr);
		assert(targetTask != nullptr);
		
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		
		bool canPropagateReadSatisfiability =
			dataAccess->readSatisfied()
			&& (dataAccess->complete() || (dataAccess->_type == READ_ACCESS_TYPE) || dataAccess->isFragment());
		bool canPropagateWriteSatisfiability =
			dataAccess->writeSatisfied()
			&& (dataAccess->complete() || dataAccess->isFragment());
		
		// Create propagation operation (if there is anything to propagate)
		if (canPropagateReadSatisfiability || canPropagateWriteSatisfiability) {
			DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
			delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS] = false;
			delayedOperation._operation[DelayedOperation::PROPAGATE_READ] = canPropagateReadSatisfiability;
			delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE] = canPropagateWriteSatisfiability;
			delayedOperation._range = dataAccess->_range;
			delayedOperation._target = targetTask;
		}
	}
	
	
	static inline void createLinkBottomMapAccessesToNextOperation(
		Task *task, DataAccessRange range, Task *next,
		/* inout */ CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(!range.empty());
		assert(next != nullptr);
		
		DelayedOperation &operation = getNewDelayedOperation(cpuDependencyData);
		operation._operation[DelayedOperation::LINK_BOTTOM_ACCESSES_TO_NEXT] = true;
		operation._target = task;
		operation._range = range;
		operation._next = next;
	}
	
	
	static inline DataAccess *linkAndCreatePropagationOperation(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccess *dataAccess, Task *task, TaskDataAccesses &accessStructures,
		TaskDataAccesses &parentAccessStructures,
		DataAccessRange range, Task *next,
		/* inout */ CPUDependencyData &cpuDependencyData
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(dataAccess->isInBottomMap());
		assert(!dataAccess->hasBeenDiscounted());
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(next != nullptr);
		
		dataAccess = fragmentAccess(
			triggererInstrumentationTaskId,
			dataAccess, range,
			accessStructures, parentAccessStructures,
			/* Consider blocking */ true
		);
		assert(dataAccess != nullptr);
		assert(dataAccess->_next == nullptr);
		
		bool wasRemovable = dataAccess->isRemovable(dataAccess->hasForcedRemoval());
		
		// Link the dataAccess access to the new task
		dataAccess->_next = next;
		
		// Remove the entry from the bottom map
		assert(dataAccess->isFragment() || (&dataAccess->_originator->getParent()->getDataAccesses() == &parentAccessStructures));
		assert(!dataAccess->isFragment() || (&dataAccess->_originator->getDataAccesses() == &parentAccessStructures));
		assert(dataAccess->isInBottomMap());
		parentAccessStructures._subaccessBottomMap.erase(dataAccess);
		dataAccess->isInBottomMap() = false;
		
		if (
			!dataAccess->complete()
			|| dataAccess->isFragment()
			|| (dataAccess->complete() && !dataAccess->hasSubaccesses())
		) {
			
			Instrument::linkedDataAccesses(
				dataAccess->_instrumentationId, next->getInstrumentationTaskId(),
				dataAccess->_range,
				true, false, triggererInstrumentationTaskId
			);
			
			createPropagationOperation(dataAccess, next, cpuDependencyData);
		} else {
			assert(dataAccess->complete());
			assert(!dataAccess->isFragment());
			assert(dataAccess->hasSubaccesses());
			
			createLinkBottomMapAccessesToNextOperation(task, dataAccess->_range.intersect(range), next, cpuDependencyData);
		}
		
		// Update the number of non-removable accesses of the dataAccess task
		if (!wasRemovable && dataAccess->isRemovable(dataAccess->hasForcedRemoval())) {
			assert(dataAccess->_next != nullptr);
			handleAccessRemoval(
				dataAccess, accessStructures, task,
				nullptr, triggererInstrumentationTaskId, /* Do not remove from bottom map */ false, cpuDependencyData
			);
		}
		
		// Return the data access since it may have been fragmented
		return dataAccess;
	}
	
	
	static inline void linkBottomMapAccessesToNext(
		Instrument::task_id_t triggererInstrumentationTaskId,
		Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range, Task *next,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(!range.empty());
		assert(next != nullptr);
		
		accessStructures._subaccessBottomMap.processIntersecting(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				DataAccess *subaccess = &(*bottomMapPosition);
				assert(subaccess != nullptr);
				assert(subaccess->isReachable());
				assert(subaccess->isInBottomMap());
				assert(!subaccess->hasBeenDiscounted());
				
				Task *subtask = subaccess->_originator;
				assert(subtask != nullptr);
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				if (subtask != task) {
					subtaskAccessStructures._lock.lock();
				} else {
					assert(subaccess->isFragment());
				}
				
				linkAndCreatePropagationOperation(
					triggererInstrumentationTaskId,
					subaccess, subtask, subtaskAccessStructures,
					accessStructures,
					range.intersect(subaccess->_range), next,
					cpuDependencyData
				);
				
				if (subtask != task) {
					subtaskAccessStructures._lock.unlock();
				}
				
				return true;
			}
		);
		
	}
	
	
	static inline void processLinkingBottomMapAccessesToNextOperation(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(delayedOperation._target != nullptr);
		assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_READ]);
		assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_WRITE]);
		assert(!delayedOperation._operation[DelayedOperation::PROPAGATE_TO_FRAGMENTS]);
		assert(delayedOperation._operation[DelayedOperation::LINK_BOTTOM_ACCESSES_TO_NEXT]);
		
		Task *task = delayedOperation._target;
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		assert(accessStructures._lock.isLockedByThisThread());
		linkBottomMapAccessesToNext(
			triggererInstrumentationTaskId,
			task, accessStructures,
			delayedOperation._range, delayedOperation._next,
			cpuDependencyData
		);
	}
	
	
	
	static inline void outerLinkMatchingInBottomMapAndCreateDelayedOperations(
		Task *task,  TaskDataAccesses &accessStructures,
		DataAccessRange range, bool weak,
		Task *parent, TaskDataAccesses &parentAccessStructures,
		/* inout */ CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!parentAccessStructures.hasBeenDeleted());
		
		// The satisfiability propagation will decrease the predecessor count as needed
		if (!weak) {
			task->increasePredecessors();
		}
		
		// Link accesses to their corresponding predecessor
		foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRange(
			parentAccessStructures,
			task, range,
			[&](DataAccess *previous) -> bool {
				assert(previous != nullptr);
				assert(previous->isReachable());
				assert(!previous->hasBeenDiscounted());
				
				Task *previousTask = previous->_originator;
				assert(previousTask != nullptr);
				
				TaskDataAccesses &previousAccessStructures = previousTask->getDataAccesses();
				assert(!previousAccessStructures.hasBeenDeleted());
				
				if (previousTask != parent) {
					previousAccessStructures._lock.lock();
				} else {
					assert(previous->isFragment());
				}
				
				previous = linkAndCreatePropagationOperation(
					task->getInstrumentationTaskId(),
					previous, previousTask, previousAccessStructures,
					parentAccessStructures,
					range.intersect(previous->_range), task,
					cpuDependencyData
				);
				
				if (previousTask != parent) {
					previousAccessStructures._lock.unlock();
				}
				
				return true;
			},
			[&](DataAccessRange missingRange) -> bool {
				assert(!parentAccessStructures._accesses.contains(missingRange));
				
				// Holes in the parent bottom map that are not in the parent accesses become fully satisfied
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock); // Need the lock since an access of data allocated in the parent may partially overlap a previous one
				accessStructures._accesses.processIntersecting(
					missingRange,
					[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
						DataAccess *targetAccess = &(*position);
						assert(targetAccess != nullptr);
						assert(!targetAccess->hasBeenDiscounted());
						
						targetAccess = fragmentAccess(
							task->getInstrumentationTaskId(),
							targetAccess, missingRange, accessStructures,
							parentAccessStructures,
							/* Consider blocking */ true
						);
						
						targetAccess->readSatisfied() = true;
						targetAccess->writeSatisfied() = true;
						if (!targetAccess->_weak) {
							task->decreasePredecessors();
						}
						
						Instrument::dataAccessBecomesSatisfied(
							targetAccess->_instrumentationId,
							true, true, false,
							task->getInstrumentationTaskId(), task->getInstrumentationTaskId()
						);
						
						return true;
					}
				);
				
				return true;
			}
			
		);
	}
	
	
	static inline void linkTaskAccesses(
		/* OUT */ CPUDependencyData &cpuDependencyData,
		Task *task
	) {
		assert(cpuDependencyData._delayedOperations.empty());
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		if (accessStructures._accesses.empty()) {
			return;
		}
		
		Task *parent = task->getParent();
		assert(parent != nullptr);
		
		TaskDataAccesses &parentAccessStructures = parent->getDataAccesses();
		assert(!parentAccessStructures.hasBeenDeleted());
		
		
		// NOTE: expensive operation that we only need if this part is instrumented
		if (sizeof(Instrument::data_access_id_t) != 0) {
			accessStructures._accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(!dataAccess->hasBeenDiscounted());
					
					dataAccess->_instrumentationId = Instrument::createdDataAccess(
						Instrument::data_access_id_t(),
						dataAccess->_type, dataAccess->_weak,
						dataAccess->_range,
						false, false, false,
						task->getInstrumentationTaskId()
					);
					
					return true;
				}
			);
		}
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> parentGuard(parentAccessStructures._lock);
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			// Create any initial missing fragments in the parent, link the previous accesses
			// and possibly some parent fragments to the new task, and create propagation
			// operations from the previous accesses to the new task.
			// 
			// The new task cannot be locked since it may have a predecessor multiple times,
			// and that could produce a dead lock if the latter is finishing (this one would
			// lock the new task first, and the predecessor later; the finishing task would
			// do the same in the reverse order). However, we need to protect the traversal
			// itself, since an already linked predecessor may produce fragmentation and thus
			// may rebalance the tree. Therefore, we just lock for advancing the iteration.
			accessStructures._accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(!dataAccess->hasBeenDiscounted());
					
					// Unlock to avoid potential deadlock
					accessStructures._lock.unlock();
					
#ifndef NDEBUG
					dataAccess->isReachable() = true;
#endif
					
					outerLinkMatchingInBottomMapAndCreateDelayedOperations(
						task, accessStructures,
						dataAccess->_range, dataAccess->_weak,
						parent, parentAccessStructures,
						cpuDependencyData
					);
					
					// Relock to advance the iterator
					accessStructures._lock.lock();
					
					return true;
				}
			);
			
			// Add the accesses to the bottom map
			accessStructures._accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(dataAccess->_next == nullptr);
					assert(dataAccess->_originator == task);
					assert(!parentAccessStructures._subaccessBottomMap.contains(dataAccess->_range));
					assert(!dataAccess->hasBeenDiscounted());
					
					parentAccessStructures._subaccessBottomMap.insert(*dataAccess);
					dataAccess->isInBottomMap() = true;
					
					return true;
				}
			);
		}
	}
	
	
	static inline bool finalizeAccess(
		Task *parent, /* INOUT */ bool &parentIsLocked,
		Task *finishedTask, DataAccess *dataAccess, DataAccessRange range,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(parent != nullptr);
		assert(finishedTask != nullptr);
		assert(dataAccess != nullptr);
		
		assert(dataAccess->_originator == finishedTask);
		assert(!dataAccess->complete());
		assert(!dataAccess->hasBeenDiscounted());
		assert(!range.empty());
		
		TaskDataAccesses &accessStructures = finishedTask->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses &parentAccessStructures = parent->getDataAccesses();
		assert(!parentAccessStructures.hasBeenDeleted());
		
		if (dataAccess->isInBottomMap()
			&& (!dataAccess->_range.fullyContainedIn(range) || dataAccess->hasForcedRemoval())
			
		) {
			// The access is in the bottom map and needs to be fragmented
			
			// Lock the parent if it is not already locked
			if (!parentIsLocked) {
				if (parentAccessStructures._lock.tryLock()) {
					parentIsLocked = true;
				} else {
					// Unlock
					accessStructures._lock.unlock();
					
					// Relock including the parent
					parentAccessStructures._lock.lock();
					accessStructures._lock.lock();
					parentIsLocked = true;
					
					// Need to restart due to locking reflow
					return false;
				}
			}
			assert(parentIsLocked);
		}
		
		// Fragment if necessary
		if (dataAccess->_range != range) {
			assert(parentIsLocked);
			dataAccess = fragmentAccess(
				finishedTask->getInstrumentationTaskId(),
				dataAccess, range,
				accessStructures, parentAccessStructures,
				/* Do not consider blocking */ false
			);
			assert(dataAccess != nullptr);
		}
		assert(dataAccess->_range == range);
		
		assert(!dataAccess->hasForcedRemoval() || (dataAccess->_next == nullptr));
		
		if (dataAccess->hasSubaccesses()) {
			// Mark the fragments as completed
			accessStructures._accessFragments.processIntersecting(
				range,
				[&](TaskDataAccesses::access_fragments_t::iterator position) -> bool {
					DataAccess *fragment = &(*position);
					assert(fragment != nullptr);
					assert(fragment->isFragment());
					assert(!fragment->hasBeenDiscounted());
					
					fragment = fragmentAccess(
						finishedTask->getInstrumentationTaskId(),
						fragment, range,
						accessStructures, accessStructures,
						/* Do not consider blocking */ false
					);
					assert(fragment != nullptr);
					
					Instrument::completedDataAccess(fragment->_instrumentationId, finishedTask->getInstrumentationTaskId());
					assert(!fragment->complete());
					fragment->complete() = true;
					
					// Update the number of non removable accesses if the fragment has become removable
					if (fragment->isRemovable(fragment->hasForcedRemoval())) {
						handleAccessRemoval(
							fragment, accessStructures, finishedTask,
							&accessStructures, finishedTask->getInstrumentationTaskId(), true, cpuDependencyData
						);
						assert(accessStructures._removalBlockers > 0);
					}
					
					return true;
				}
			);
			
			if (dataAccess->_next != nullptr) {
				// Link bottom map subaccesses to the next of the current access and remove them from the bottom map
				linkBottomMapAccessesToNext(
					finishedTask->getInstrumentationTaskId(),
					dataAccess->_originator, accessStructures, dataAccess->_range,
					dataAccess->_next, /* OUT */ cpuDependencyData
				);
			}
		}
		
		// Mark it as complete
		Instrument::completedDataAccess(dataAccess->_instrumentationId, finishedTask->getInstrumentationTaskId());
		assert(!dataAccess->complete());
		dataAccess->complete() = true;
		
		if (!dataAccess->hasSubaccesses() && (dataAccess->_next != nullptr) && dataAccess->writeSatisfied()) {
			createPropagationOperation(dataAccess, dataAccess->_next, /* OUT */ cpuDependencyData);
		}
		
		// Handle propagation of forced removal of accesses
		if (dataAccess->hasForcedRemoval() && dataAccess->hasSubaccesses()) {
			activateForcedRemovalOfBottomMapAccesses(
				finishedTask, accessStructures,
				dataAccess->_range,
				cpuDependencyData, finishedTask->getInstrumentationTaskId()
			);
		}
			
		// Update the number of non removable accesses of the task
		if (dataAccess->isRemovable(dataAccess->hasForcedRemoval())) {
			handleAccessRemoval(
				dataAccess, accessStructures, finishedTask,
				&parentAccessStructures, finishedTask->getInstrumentationTaskId(), true, cpuDependencyData
			);
		}
		
		// No need to reflow due to relock
		return true;
	}
	
	
	static void handleRemovableTasks(
		/* inout */ CPUDependencyData::removable_task_list_t &removableTasks,
		CPU *cpu,
		WorkerThread *thread
	) {
		for (Task *removableTask : removableTasks) {
			TaskFinalization::disposeOrUnblockTask(removableTask, cpu, thread);
		}
		removableTasks.clear();
	}
	
	static inline CPUDependencyData &getCPUDependencyDataCPUAndThread(/* out */ CPU * &cpu, /* out */ WorkerThread * &thread)
	{
		thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		cpu = thread->getHardwarePlace();
		assert(cpu != nullptr);
		
		return cpu->_dependencyData;
	}
	
	
	
public:

	//! \brief creates a task data access taking into account repeated accesses but does not link it to previous accesses nor superaccesses
	//! 
	//! \param[inout] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] weak true iff the access is weak
	//! \param[in] range the range of data covered by the access
	static inline void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange range
	) {
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		accessStructures._accesses.fragmentIntersecting(
			range,
			[&](DataAccess const &toBeDuplicated) -> DataAccess * {
				return duplicateDataAccess(toBeDuplicated, accessStructures, false);
			},
			[](DataAccess *, DataAccess *) {}
		);
		
		accessStructures._accesses.processIntersectingAndMissing(
			range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *oldAccess = &(*position);
				assert(oldAccess != nullptr);
				
				upgradeAccess(oldAccess, accessType, weak);
				
				return true;
			},
			[&](DataAccessRange missingRange) -> bool {
				DataAccess *newAccess = createAccess(task, accessType, weak, missingRange, false);
				
				accessStructures._removalBlockers++;
				accessStructures._accesses.insert(*newAccess);
				
				return true;
			}
		);
	}
	
	
	//! \brief Performs the task dependency registration procedure
	//! 
	//! \param[in] task the Task whose dependencies need to be calculated
	//! 
	//! \returns true if the task is already ready
	static inline bool registerTaskDataAccesses(Task *task)
	{
		assert(task != 0);
		
		nanos_task_info *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		
		// This part creates the DataAccesses and calculates any possible upgrade
		taskInfo->register_depinfo(task, task->getArgsBlock());
		
		if (!task->getDataAccesses()._accesses.empty()) {
			// The blocking count is decreased once all the accesses become removable
			task->increaseRemovalBlockingCount();
			
			task->increasePredecessors(2);
			
			WorkerThread *currentThread = nullptr;
			CPU *cpu = nullptr;
			
			currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);
			cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			
			CPUDependencyData &cpuDependencyData = cpu->_dependencyData;
			
#ifndef NDEBUG
			{
				bool alreadyTaken = false;
				assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
			}
#endif
			
			// This part actually inserts the accesses into the dependency system
			linkTaskAccesses(cpuDependencyData, task);
			
			processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
				task->getInstrumentationTaskId(),
				cpuDependencyData, cpu, currentThread
			);
			
#ifndef NDEBUG
			{
				bool alreadyTaken = true;
				assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
			}
#endif
			
			return task->decreasePredecessors(2);
		} else {
			return true;
		}
	}
	
	
	static inline void unregisterTaskDataAccesses(Task *task)
	{
		assert(task != 0);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		
		if (accesses.empty()) {
			return;
		}
		
		CPU *cpu = nullptr;
		WorkerThread *currentThread = nullptr;
		CPUDependencyData &cpuDependencyData = getCPUDependencyDataCPUAndThread(/* out */ cpu, /* out */ currentThread);
		assert(cpu != nullptr);
		assert(currentThread != nullptr);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		
		Task *parent = task->getParent();
		assert(parent != nullptr);
		
		assert(cpuDependencyData._delayedOperations.empty());
		
		{
			bool parentIsLocked = false;
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.processAllWithRestart(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					
					// Returns false if the lock has been droped to also lock the parent.
					// This allows the traversal to restart from the equivalent point, since the contents of
					// accesses may have changed during the relocking operation (due to fragmentation).
					return finalizeAccess(
						parent, /* INOUT */ parentIsLocked,
						task, dataAccess, dataAccess->_range,
						/* OUT */ cpuDependencyData
					);
				}
			);
			
			if (parentIsLocked) {
				TaskDataAccesses &parentAccessStructures = parent->getDataAccesses();
				assert(!parentAccessStructures.hasBeenDeleted());
				parentAccessStructures._lock.unlock();
			}
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
			task->getInstrumentationTaskId(),
			cpuDependencyData, cpu, currentThread
		);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}
	
	
	static void handleEnterBlocking(Task *task)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		if (!accessStructures._accesses.empty()) {
			task->decreaseRemovalBlockingCount();
		}
	}
	
	
	static void handleExitBlocking(Task *task)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		if (!accessStructures._accesses.empty()) {
			task->increaseRemovalBlockingCount();
		}
	}
	
	
	static void handleEnterTaskwait(Task *task)
	{
		assert(task != nullptr);
		
		CPU *cpu = nullptr;
		WorkerThread *currentThread = nullptr;
		CPUDependencyData &cpuDependencyData = getCPUDependencyDataCPUAndThread(/* out */ cpu, /* out */ currentThread);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		
		{
			TaskDataAccesses &accessStructures = task->getDataAccesses();
			assert(!accessStructures.hasBeenDeleted());
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			if (!accessStructures._accesses.empty()) {
				assert(accessStructures._removalBlockers > 0);
				task->decreaseRemovalBlockingCount();
			}
			
			activateForcedRemovalOfBottomMapAccesses(task, accessStructures, cpuDependencyData, task->getInstrumentationTaskId());
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
			task->getInstrumentationTaskId(),
			cpuDependencyData, cpu, currentThread
		);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}
	
	
	static void handleExitTaskwait(Task *task)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		
		// In principle, all inner tasks must have ended
		
		assert(accessStructures._subaccessBottomMap.empty());
		
		if (!accessStructures._accesses.empty()) {
			assert(accessStructures._removalBlockers > 0);
			task->increaseRemovalBlockingCount();
			
			// Mark all accesses as not having subaccesses
			accessStructures._accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(!dataAccess->hasBeenDiscounted());
					
					dataAccess->hasSubaccesses() = false;
					
					return true;
				}
			);
			
			// Delete all fragments
			accessStructures._accessFragments.processAll(
				[&](TaskDataAccesses::access_fragments_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(dataAccess->isFragment());
					assert(accessStructures._removalBlockers > 0);
					
					accessStructures._removalBlockers--;
					assert(accessStructures._removalBlockers >= 0);
					
					Instrument::removedDataAccess(dataAccess->_instrumentationId, task->getInstrumentationTaskId());
					accessStructures._accessFragments.erase(dataAccess);
					delete dataAccess;
					
					return true;
				}
			);
			accessStructures._accessFragments.clear();
		}
	}
	
};


#endif // DATA_ACCESS_REGISTRATION_HPP
