#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <cassert>
#include <deque>
#include <mutex>
#include <vector>

#include "BottomMapEntry.hpp"
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
	
	
	static inline BottomMapEntry *fragmentBottomMapEntry(
		BottomMapEntry *bottomMapEntry, DataAccessRange range,
		TaskDataAccesses &accessStructures
	) {
		if (bottomMapEntry->_range.fullyContainedIn(range)) {
			// Nothing to fragment
			return bottomMapEntry;
		}
		
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		TaskDataAccesses::subaccess_bottom_map_t::iterator position =
			accessStructures._subaccessBottomMap.iterator_to(*bottomMapEntry);
		position = accessStructures._subaccessBottomMap.fragmentByIntersection(
			position, range,
			false,
			[&](BottomMapEntry const &toBeDuplicated) -> BottomMapEntry * {
				return new BottomMapEntry(DataAccessRange(), toBeDuplicated._task, toBeDuplicated._local);
			},
			[&](__attribute__((unused)) BottomMapEntry *fragment, __attribute__((unused)) BottomMapEntry *originalBottomMapEntry) {
			}
		);
		
		bottomMapEntry = &(*position);
		assert(bottomMapEntry != nullptr);
		assert(bottomMapEntry->_range.fullyContainedIn(range));
		
		return bottomMapEntry;
	}
	
	
	static inline DataAccess *fragmentAccess(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccess *dataAccess, DataAccessRange range,
		TaskDataAccesses &accessStructures,
		bool considerTaskBlocking
	) {
		assert(dataAccess != nullptr);
		// assert(accessStructures._lock.isLockedByThisThread()); // Not necessary when fragmenting an access that is not reachable
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		assert(&dataAccess->_originator->getDataAccesses() == &accessStructures);
		assert(!accessStructures.hasBeenDeleted());
		assert(!dataAccess->hasBeenDiscounted());
		
		if (dataAccess->_range.fullyContainedIn(range)) {
			// Nothing to fragment
			return dataAccess;
		}
		
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
						if (fragment->isTopmost()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "T", "Topmost", triggererInstrumentationTaskId);
						}
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
						if (fragment->isTopmost()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "T", "Topmost", triggererInstrumentationTaskId);
						}
					} else {
						Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range, triggererInstrumentationTaskId);
					}
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->_range.fullyContainedIn(range));
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
		Instrument::task_id_t triggererInstrumentationTaskId,
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
		bool propagateReadSatisfiability = delayedOperation._propagateRead;
		bool propagateWriteSatisfiability = delayedOperation._propagateWrite;
		Task *targetTask = delayedOperation._target;
		DataAccessRange range = delayedOperation._range;
		
		assert(targetTask != nullptr);
		
		TaskDataAccesses &targetTaskAccessStructures = targetTask->getDataAccesses();
		assert(!targetTaskAccessStructures.hasBeenDeleted());
		assert(targetTaskAccessStructures._lock.isLockedByThisThread());
		
		// NOTE: An access is discounted before traversing the fragments, so by the time we reach this point, the counter could be 0
		
		targetTaskAccessStructures._accessFragments.processIntersecting(
			range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *targetAccess = &(*position);
				assert(targetAccess != nullptr);
				assert(targetAccess->isFragment());
				assert(targetAccess->isReachable());
				assert(targetAccess->_originator == targetTask);
				assert(!targetAccess->hasBeenDiscounted());
				
				__attribute__((unused)) bool acceptsReadSatisfiability = !targetAccess->readSatisfied();
				__attribute__((unused)) bool acceptsWriteSatisfiability = !targetAccess->writeSatisfied();
				
				// Fragments always get the same satisfiability changes as their corresponding accesses
				assert(!propagateReadSatisfiability || acceptsReadSatisfiability);
				assert(!propagateWriteSatisfiability || acceptsWriteSatisfiability);
				
				// Fragment if necessary
				if (!targetAccess->_range.fullyContainedIn(range)) {
					targetAccess = fragmentAccess(
						triggererInstrumentationTaskId,
						targetAccess, range,
						targetTaskAccessStructures,
						/* Do not affect originator blocking counter */ false
					);
					assert(targetAccess != nullptr);
					assert(targetAccess->_range.fullyContainedIn(range));
				}
				
				bool wasRemovable = targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				
				assert(targetAccess->readSatisfied() || !targetAccess->writeSatisfied());
				
				// Update the satisfiability
				if (propagateReadSatisfiability) {
					assert(!targetAccess->readSatisfied());
					targetAccess->readSatisfied() = true;
				}
				if (propagateWriteSatisfiability) {
					assert(!targetAccess->writeSatisfied());
					targetAccess->writeSatisfied() = true;
				}
				
				assert(targetAccess->readSatisfied() || !targetAccess->writeSatisfied());
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->_instrumentationId,
					propagateReadSatisfiability,
					propagateWriteSatisfiability,
					false,
					triggererInstrumentationTaskId,
					targetTask->getInstrumentationTaskId()
				);
				
				// Update the number of non removable accesses of the task
				bool becomesRemovable = !wasRemovable && targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				if (becomesRemovable) {
					handleAccessRemoval(
						targetAccess, targetTaskAccessStructures, targetTask,
						triggererInstrumentationTaskId, cpuDependencyData
					);
				}
				
				assert((targetAccess->_next != nullptr) || targetAccess->isInBottomMap());
				
				// Propagates as is to subaccesses, and as a regular access to outer accesses
				bool propagationToSubtask =
					(targetAccess->_next != nullptr) && (targetAccess->_next->getParent() == targetTask);
				
				bool canPropagateReadSatisfiability = propagateReadSatisfiability;
				bool canPropagateWriteSatisfiability = propagateWriteSatisfiability;
				
				if (propagationToSubtask) {
					// The next of a fragment (of a direct subtask) inherits the same satisfiability as the fragments
				} else {
					canPropagateReadSatisfiability &= targetAccess->readSatisfied()
						&& (targetAccess->complete() || (targetAccess->_type == READ_ACCESS_TYPE));
					canPropagateWriteSatisfiability &= targetAccess->writeSatisfied()
						&& targetAccess->complete();
				}
				
				Task *nextTask = targetAccess->_next;
				
				// The next can only become topmost if it belongs to the same dependency domain
				bool makesNextTopmost = becomesRemovable && (nextTask != nullptr) && (nextTask->getParent() == targetTask);
				
				// Continue to next iteration if there is nothing to propagate
				if (
					!canPropagateReadSatisfiability
					&& !canPropagateWriteSatisfiability
					&& !makesNextTopmost
				) {
					return true;
				}
				
				if (nextTask != nullptr) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
					DelayedOperation nextOperation;
#else
					DelayedOperation &nextOperation = getNewDelayedOperation(cpuDependencyData);
					nextOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
#endif
					nextOperation._propagateRead = canPropagateReadSatisfiability;
					nextOperation._propagateWrite = canPropagateWriteSatisfiability;
					nextOperation._makeTopmost = makesNextTopmost;
					nextOperation._range = targetAccess->_range;
					nextOperation._target = targetAccess->_next;
					
#if NO_DEPENDENCY_DELAYED_OPERATIONS
					TaskDataAccesses &nextTaskAccessStructures = nextTask->getDataAccesses();
					std::lock_guard<TaskDataAccesses::spinlock_t> guard(nextTaskAccessStructures._lock);
					
					propagateSatisfiabilityPlain(triggererInstrumentationTaskId, nextOperation, cpuDependencyData);
#endif
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
		bool propagateReadSatisfiability = delayedOperation._propagateRead;
		bool propagateWriteSatisfiability = delayedOperation._propagateWrite;
		bool makeTopmost = delayedOperation._makeTopmost;
		Task *targetTask = delayedOperation._target;
		DataAccessRange range = delayedOperation._range;
		
		assert(targetTask != nullptr);
		
		TaskDataAccesses &targetTaskAccessStructures = targetTask->getDataAccesses();
		assert(!targetTaskAccessStructures.hasBeenDeleted());
		assert(targetTaskAccessStructures._lock.isLockedByThisThread());
		
		targetTaskAccessStructures._accesses.processIntersecting(
			range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *targetAccess = &(*position);
				assert(targetAccess != nullptr);
				assert(targetAccess->isReachable());
				assert(targetAccess->_originator == targetTask);
				assert(!targetAccess->hasBeenDiscounted());
				assert(!targetAccess->isTopmost() || !makeTopmost);
				
				bool acceptsReadSatisfiability = !targetAccess->readSatisfied();
				bool acceptsWriteSatisfiability = !targetAccess->writeSatisfied();
				
				// Skip accesses whose state does not change
				if (
					(!propagateReadSatisfiability || !acceptsReadSatisfiability)
					&& (!propagateWriteSatisfiability || !acceptsWriteSatisfiability)
					&& !makeTopmost
				) {
					return true;
				}
				
				// Fragment if necessary
				if (!targetAccess->_range.fullyContainedIn(range) || targetAccess->hasForcedRemoval()) {
					targetAccess = fragmentAccess(
						triggererInstrumentationTaskId,
						targetAccess, range, targetTaskAccessStructures,
						/* Affect originator blocking counter */ true
					);
					assert(targetAccess != nullptr);
				}
				
				bool wasSatisfied = targetAccess->satisfied();
				bool wasRemovable = targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				
				assert(targetAccess->readSatisfied() || !targetAccess->writeSatisfied());
				
				// Update the satisfiability
				if (propagateReadSatisfiability && acceptsReadSatisfiability) {
					assert(!targetAccess->readSatisfied());
					targetAccess->readSatisfied() = true;
				}
				if (propagateWriteSatisfiability && acceptsWriteSatisfiability) {
					assert(!targetAccess->writeSatisfied());
					targetAccess->writeSatisfied() = true;
				}
				if (makeTopmost) {
					targetAccess->isTopmost() = true;
					Instrument::newDataAccessProperty(targetAccess->_instrumentationId, "T", "Topmost", triggererInstrumentationTaskId);
				}
				
				assert(targetAccess->readSatisfied() || !targetAccess->writeSatisfied());
				
				// Update the number of non removable accesses of the task
				bool becomesRemovable = !wasRemovable && targetAccess->isRemovable(targetAccess->hasForcedRemoval());
				if (becomesRemovable) {
					handleAccessRemoval(
						targetAccess, targetTaskAccessStructures, targetTask,
						triggererInstrumentationTaskId, cpuDependencyData
					);
				}
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->_instrumentationId,
					propagateReadSatisfiability && acceptsReadSatisfiability,
					propagateWriteSatisfiability && acceptsWriteSatisfiability,
					false,
					triggererInstrumentationTaskId,
					targetTask->getInstrumentationTaskId()
				);
				
				
				// If the target access becomes satisfied decrease the predecessor count of the task
				// If it becomes 0 then add it to the list of satisfied originators
				if (!targetAccess->_weak && !wasSatisfied && targetAccess->satisfied()) {
					if (targetTask->decreasePredecessors()) {
						cpuDependencyData._satisfiedOriginators.push_back(targetTask);
					}
				}
				
				Task *nextTask = targetAccess->_next;
				
				// The next can only become topmost if it belongs to the same dependency domain
				bool makesNextTopmost = becomesRemovable && (nextTask != nullptr) && (nextTask->getParent() == targetTask->getParent());
				
				if (targetAccess->hasSubaccesses()) {
					// Propagate to fragments
					assert(targetAccess->_range.fullyContainedIn(range));
					
					// Only propagate to fragments if there is satisfiability to propagate.
					// The topmost property is internal to the inner dependency domain.
					// Otherwise we may end up accessing a fragment that has already been
					// discounted.
					if (
						(propagateReadSatisfiability && acceptsReadSatisfiability)
						|| (propagateWriteSatisfiability && acceptsWriteSatisfiability)
					) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						DelayedOperation delayedOperation;
#else
						DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
						delayedOperation._operationType = DelayedOperation::propagate_satisfiability_to_fragments_operation;
#endif
						delayedOperation._propagateRead = propagateReadSatisfiability && acceptsReadSatisfiability;
						delayedOperation._propagateWrite = propagateWriteSatisfiability && acceptsWriteSatisfiability;
						delayedOperation._range = targetAccess->_range;
						delayedOperation._target = targetTask;
						
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						propagateSatisfiabilityToFragments(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
#endif
					}
					
					// Propagate topmost property to next
					if (makesNextTopmost) {
						makeRangeTopmost(
							triggererInstrumentationTaskId,
							targetAccess->_range, nextTask,
							cpuDependencyData
						);
					}
				} else if (nextTask != nullptr) {
					assert(!targetAccess->hasSubaccesses());
					
					// Propagate to next
					bool canPropagateReadSatisfiability =
						propagateReadSatisfiability
						&& targetAccess->readSatisfied()
						&& (targetAccess->complete() || (targetAccess->_type == READ_ACCESS_TYPE));
					bool canPropagateWriteSatisfiability =
						propagateWriteSatisfiability
						&& targetAccess->writeSatisfied() && (targetAccess->complete());
					
					if (canPropagateReadSatisfiability || canPropagateWriteSatisfiability || makesNextTopmost) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						DelayedOperation delayedOperation;
#else
						DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
						delayedOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
#endif
						delayedOperation._propagateRead = canPropagateReadSatisfiability;
						delayedOperation._propagateWrite = canPropagateWriteSatisfiability;
						delayedOperation._makeTopmost = makesNextTopmost;
						delayedOperation._range = targetAccess->_range;
						delayedOperation._target = nextTask;
						
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						TaskDataAccesses &nextTaskAccessStructures = nextTask->getDataAccesses();
						std::lock_guard<TaskDataAccesses::spinlock_t> guard(nextTaskAccessStructures._lock);
						propagateSatisfiabilityPlain(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
#endif
					}
				}
				
				return true;
			}
		);
	}
	
	
	static inline void activateForcedRemovalOfBottomMapAccesses(
		Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range,
		/* OUT */ CPUDependencyData &cpuDependencyData,
		Instrument::task_id_t triggererInstrumentationTaskId
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		// For each bottom map entry
		accessStructures._subaccessBottomMap.processIntersecting(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
				BottomMapEntry *bottomMapEntry = &(*position);
				assert(bottomMapEntry != nullptr);
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->_range);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				if (subtask != task) {
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					subtaskAccessStructures._accesses.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *dataAccess = &(*accessPosition);
							
							assert(dataAccess->_next == nullptr);
							assert(dataAccess->isInBottomMap());
							assert(!dataAccess->hasBeenDiscounted());
							
							assert(!dataAccess->hasForcedRemoval());
							
							dataAccess = fragmentAccess(
								triggererInstrumentationTaskId,
								dataAccess, subrange, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(dataAccess->_next == nullptr);
							dataAccess->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal", triggererInstrumentationTaskId);
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, cpuDependencyData, triggererInstrumentationTaskId);
							}
							
							if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(
									dataAccess, subtaskAccessStructures, subtask,
									triggererInstrumentationTaskId, cpuDependencyData
								);
							}
							
							return true;
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					accessStructures._accessFragments.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->_next == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(
								triggererInstrumentationTaskId,
								fragment, subrange, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(fragment->_next == nullptr);
							fragment->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "F", "Forced Removal", triggererInstrumentationTaskId);
							
							if (!fragment->isRemovable(false) && fragment->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(
									fragment, accessStructures, task,
									triggererInstrumentationTaskId, cpuDependencyData
								);
							}
							
							return true;
						}
					);
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
		
		// For each bottom map entry
		accessStructures._subaccessBottomMap.processAll(
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
				BottomMapEntry *bottomMapEntry = &(*position);
				assert(bottomMapEntry != nullptr);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				if (subtask != task) {
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					subtaskAccessStructures._accesses.processIntersecting(
						bottomMapEntry->_range,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *dataAccess = &(*accessPosition);
							
							assert(dataAccess->_next == nullptr);
							assert(dataAccess->isInBottomMap());
							assert(!dataAccess->hasBeenDiscounted());
							
							assert(!dataAccess->hasForcedRemoval());
							
							dataAccess = fragmentAccess(
								triggererInstrumentationTaskId,
								dataAccess, bottomMapEntry->_range, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(dataAccess->_next == nullptr);
							dataAccess->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal", triggererInstrumentationTaskId);
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, cpuDependencyData, triggererInstrumentationTaskId);
							}
							
							if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(
									dataAccess, subtaskAccessStructures, subtask,
									triggererInstrumentationTaskId, cpuDependencyData
								);
							}
							
							return true;
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					accessStructures._accessFragments.processIntersecting(
						bottomMapEntry->_range,
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->_next == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							assert(!fragment->hasForcedRemoval());
							
							fragment = fragmentAccess(
								triggererInstrumentationTaskId,
								fragment, bottomMapEntry->_range, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(fragment->_next == nullptr);
							fragment->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "F", "Forced Removal", triggererInstrumentationTaskId);
							
							if (!fragment->isRemovable(false) && fragment->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(
									fragment, accessStructures, task,
									triggererInstrumentationTaskId, cpuDependencyData
								);
							}
							
							return true;
						}
					);
				}
				
				// Remove the bottom map entry
				accessStructures._subaccessBottomMap.erase(bottomMapEntry);
				delete bottomMapEntry;
				
				return true;
			}
		);
	}
	
	
	static void processDelayedOperation(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		switch (delayedOperation._operationType) {
			case DelayedOperation::link_bottom_map_accesses_operation:
				linkBottomMapAccessesToNext(
					triggererInstrumentationTaskId,
					delayedOperation,
					cpuDependencyData
				);
				break;
			case DelayedOperation::propagate_satisfiability_to_fragments_operation:
				propagateSatisfiabilityToFragments(
					triggererInstrumentationTaskId,
					delayedOperation,
					cpuDependencyData
				);
				break;
			case DelayedOperation::propagate_satisfiability_plain_operation:
				propagateSatisfiabilityPlain(
					triggererInstrumentationTaskId,
					delayedOperation,
					cpuDependencyData
				);
				break;
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
#if NO_DEPENDENCY_DELAYED_OPERATIONS
#else
		processDelayedOperations(triggererInstrumentationTaskId, cpuDependencyData);
#endif
		
		processSatisfiedOriginators(cpuDependencyData, cpu);
		assert(cpuDependencyData._satisfiedOriginators.empty());
		
		handleRemovableTasks(cpuDependencyData._removableTasks, cpu, currentThread);
	}
	
	
	static inline DataAccess *createInitialFragment(
		Instrument::task_id_t triggererInstrumentationTaskId,
		TaskDataAccesses::accesses_t::iterator accessPosition,
		TaskDataAccesses &accessStructures,
		DataAccessRange subrange,
		bool createSubrangeBottomMapEntry, /* Out */ BottomMapEntry *&bottomMapEntry
	) {
		DataAccess *dataAccess = &(*accessPosition);
		assert(dataAccess != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(bottomMapEntry == nullptr);
		
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
		
		assert(dataAccess->readSatisfied() || !dataAccess->writeSatisfied());
		assert(fragment->readSatisfied() || !fragment->writeSatisfied());
			
		fragment->readSatisfied() = dataAccess->readSatisfied();
		fragment->writeSatisfied() = dataAccess->writeSatisfied();
		fragment->complete() = dataAccess->complete();
#ifndef NDEBUG
		fragment->isReachable() = true;
#endif
		
		assert(fragment->readSatisfied() || !fragment->writeSatisfied());
		
		accessStructures._accessFragments.insert(*fragment);
		fragment->isInBottomMap() = true;
		dataAccess->hasSubaccesses() = true;
		
		if (createSubrangeBottomMapEntry) {
			bottomMapEntry = new BottomMapEntry(dataAccess->_range, dataAccess->_originator, /* Not local */ false);
			accessStructures._subaccessBottomMap.insert(*bottomMapEntry);
		} else if (subrange != dataAccess->_range) {
			dataAccess->_range.processIntersectingFragments(
				subrange,
				[&](DataAccessRange excludedSubrange) {
					bottomMapEntry = new BottomMapEntry(excludedSubrange, dataAccess->_originator, /* Not local */ false);
					accessStructures._subaccessBottomMap.insert(*bottomMapEntry);
				},
				[&](__attribute__((unused)) DataAccessRange intersection) {
					assert(!createSubrangeBottomMapEntry);
				},
				[&](__attribute__((unused)) DataAccessRange unmatchedRange) {
					// This part is not covered by the access
				}
			);
		}
		
		// Fragments also participate in the counter of non removable accesses
		accessStructures._removalBlockers++;
		
		return fragment;
	}
	
	
	template <typename MatchingProcessorType, typename MissingProcessorType>
	static inline bool foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRange(
		Task *parent, TaskDataAccesses &parentAccessStructures,
		DataAccessRange range, Task *task,
		MatchingProcessorType matchingProcessor, MissingProcessorType missingProcessor,
		bool removeBottomMapEntry
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert((&parentAccessStructures) == (&parent->getDataAccesses()));
		assert(!parentAccessStructures.hasBeenDeleted());
		
		return parentAccessStructures._subaccessBottomMap.processIntersectingAndMissing(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->_range);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				bool result = true;
				if (subtask != parent) {
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					result = subtaskAccessStructures._accesses.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *previous = &(*accessPosition);
							
							assert(previous->_next == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(
								task->getInstrumentationTaskId(),
								previous, subrange, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							return matchingProcessor(previous, bottomMapEntry);
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					
					// For each fragment of the parent that matches
					result = parentAccessStructures._accessFragments.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::accesses_t::iterator fragmentPosition) -> bool {
							DataAccess *previous = &(*fragmentPosition);
							
							assert(previous->_next == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(
								task->getInstrumentationTaskId(),
								previous, subrange, parentAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							return matchingProcessor(previous, bottomMapEntry);
						}
					);
				}
				
				if (removeBottomMapEntry) {
					bottomMapEntry = fragmentBottomMapEntry(bottomMapEntry, subrange, parentAccessStructures);
					parentAccessStructures._subaccessBottomMap.erase(*bottomMapEntry);
					delete bottomMapEntry;
				}
				
				return result;
			},
			[&](DataAccessRange missingRange) -> bool {
				parentAccessStructures._accesses.processIntersectingAndMissing(
					missingRange,
					[&](TaskDataAccesses::accesses_t::iterator superaccessPosition) -> bool {
						BottomMapEntry *bottomMapEntry = nullptr;
						
						DataAccess *previous = createInitialFragment(
							task->getInstrumentationTaskId(),
							superaccessPosition, parentAccessStructures,
							missingRange, !removeBottomMapEntry, /* Out */ bottomMapEntry
						);
						assert(previous != nullptr);
						
						previous->isTopmost() = true;
						Instrument::newDataAccessProperty(previous->_instrumentationId, "T", "Topmost", task->getInstrumentationTaskId());
						
						assert(previous->isFragment());
						previous = fragmentAccess(
							task->getInstrumentationTaskId(),
							previous, missingRange, parentAccessStructures,
							/* Affect originator blocking counter */ true
						);
						
						return matchingProcessor(previous, bottomMapEntry);
					},
					[&](DataAccessRange rangeUncoveredByParent) -> bool {
						return missingProcessor(rangeUncoveredByParent);
					}
				);
				
				return true;
			}
		);
	}
	
	
	static inline void propagateSatisfiability(
		__attribute__((unused)) Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccess *dataAccess, Task *targetTask,
		bool makeTopmost,
		/* inout */ CPUDependencyData &cpuDependencyData,
		bool delayable = true
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
		
		if (canPropagateReadSatisfiability || canPropagateWriteSatisfiability || makeTopmost) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
			DelayedOperation delayedOperation;
			delayedOperation._propagateRead = canPropagateReadSatisfiability;
			delayedOperation._propagateWrite = canPropagateWriteSatisfiability;
			delayedOperation._makeTopmost = makeTopmost;
			delayedOperation._range = dataAccess->_range;
			delayedOperation._target = targetTask;
			
			TaskDataAccesses &targetAccessStructures = targetTask->getDataAccesses();
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(targetAccessStructures._lock);
			
			propagateSatisfiabilityPlain(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
#else
			if (delayable) {
				DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
				
				delayedOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
				delayedOperation._propagateRead = canPropagateReadSatisfiability;
				delayedOperation._propagateWrite = canPropagateWriteSatisfiability;
				delayedOperation._makeTopmost = makeTopmost;
				delayedOperation._range = dataAccess->_range;
				delayedOperation._target = targetTask;
			} else {
				DelayedOperation delayedOperation;
				delayedOperation._propagateRead = canPropagateReadSatisfiability;
				delayedOperation._propagateWrite = canPropagateWriteSatisfiability;
				delayedOperation._makeTopmost = makeTopmost;
				delayedOperation._range = dataAccess->_range;
				delayedOperation._target = targetTask;
				
				TaskDataAccesses &targetAccessStructures = targetTask->getDataAccesses();
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(targetAccessStructures._lock);
				
				propagateSatisfiabilityPlain(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
			}
#endif
		}
	}
	
	
	static inline void makeRangeTopmost(
		__attribute__((unused)) Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccessRange range, Task *task,
		/* inout */ CPUDependencyData &cpuDependencyData,
		bool delayable = true
	) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
		DelayedOperation delayedOperation;
		
		delayedOperation._propagateRead = false;
		delayedOperation._propagateWrite = false;
		delayedOperation._makeTopmost = true;
		delayedOperation._range = range;
		delayedOperation._target = task;
		
		TaskDataAccesses &targetAccessStructures = task->getDataAccesses();
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(targetAccessStructures._lock);
		propagateSatisfiabilityPlain(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
#else
		if (delayable) {
			DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
			delayedOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
			
			delayedOperation._propagateRead = false;
			delayedOperation._propagateWrite = false;
			delayedOperation._makeTopmost = true;
			delayedOperation._range = range;
			delayedOperation._target = task;
		} else {
			DelayedOperation delayedOperation;
			
			delayedOperation._propagateRead = false;
			delayedOperation._propagateWrite = false;
			delayedOperation._makeTopmost = true;
			delayedOperation._range = range;
			delayedOperation._target = task;
			
			TaskDataAccesses &targetAccessStructures = task->getDataAccesses();
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(targetAccessStructures._lock);
			propagateSatisfiabilityPlain(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
		}
#endif
	}
	
	
	static inline DataAccess *linkAndPropagate(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DataAccess *dataAccess, Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range, Task *next,
		bool propagateTopmost,
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
			accessStructures,
			/* Consider blocking */ true
		);
		assert(dataAccess != nullptr);
		assert(dataAccess->_next == nullptr);
		
		assert(!dataAccess->hasForcedRemoval());
		assert(!dataAccess->isRemovable(false));
		
		// Link the dataAccess access to the new task
		dataAccess->_next = next;
		dataAccess->isInBottomMap() = false;
		
		bool becomesRemovable = dataAccess->isRemovable(false);
		
		bool makesNextTopmost = propagateTopmost && becomesRemovable;
		if (makesNextTopmost) {
			if (dataAccess->isFragment()) {
				// Child after fragment
				makesNextTopmost = (next->getParent() == dataAccess->_originator);
			} else {
				// Potentially sibling tasks
				makesNextTopmost = (next->getParent() == task->getParent());
			}
		}
		
		Instrument::linkedDataAccesses(
			dataAccess->_instrumentationId, next->getInstrumentationTaskId(),
			dataAccess->_range,
			true, false, triggererInstrumentationTaskId
		);
		
		if (
			!dataAccess->complete()
			|| dataAccess->isFragment()
			|| (dataAccess->complete() && !dataAccess->hasSubaccesses())
		) {
			
			propagateSatisfiability(
				triggererInstrumentationTaskId,
				dataAccess, next,
				makesNextTopmost,
				cpuDependencyData,
				false /* Cannot be delayed */
			);
		} else {
			assert(dataAccess->complete());
			assert(!dataAccess->isFragment());
			assert(dataAccess->hasSubaccesses());
			
			if (makesNextTopmost) {
				makeRangeTopmost(
					triggererInstrumentationTaskId,
					dataAccess->_range.intersect(range), next,
					cpuDependencyData,
					false /* Cannot be delayed */
				);
			}
			
			// This operation cannot be delayed since otherwise there could be update races
			DelayedOperation delayedOperation;
			delayedOperation._next = next;
			delayedOperation._range = dataAccess->_range.intersect(range);
			delayedOperation._target = task;
			linkBottomMapAccessesToNext(triggererInstrumentationTaskId, delayedOperation, cpuDependencyData);
		}
		
		// Update the number of non-removable accesses of the dataAccess task
		if (becomesRemovable) {
			assert(dataAccess->_next != nullptr);
			handleAccessRemoval(
				dataAccess, accessStructures, task,
				triggererInstrumentationTaskId, cpuDependencyData
			);
		}
		
		// Return the data access since it may have been fragmented
		return dataAccess;
	}
	
	
	static inline void linkBottomMapAccessesToNext(
		Instrument::task_id_t triggererInstrumentationTaskId,
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		DataAccessRange range = delayedOperation._range;
		Task *task = delayedOperation._target;
		Task *next = delayedOperation._next;
		
		assert(task != nullptr);
		assert(!range.empty());
		assert(next != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		accessStructures._subaccessBottomMap.processIntersecting(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->_range);
				
				if (subtask != task) {
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					subtaskAccessStructures._accesses.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *subaccess = &(*accessPosition);
							assert(subaccess != nullptr);
							assert(subaccess->isReachable());
							assert(subaccess->_next == nullptr);
							assert(subaccess->isInBottomMap());
							assert(!subaccess->hasBeenDiscounted());
							
							subaccess = fragmentAccess(
								triggererInstrumentationTaskId,
								subaccess, subrange, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							linkAndPropagate(
								triggererInstrumentationTaskId,
								subaccess, subtask, subtaskAccessStructures,
								subrange.intersect(subaccess->_range), next,
								/* Propagate topmost property */ true,
								cpuDependencyData
							);
							
							return true;
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					accessStructures._accessFragments.processIntersecting(
						subrange,
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->_next == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(
								triggererInstrumentationTaskId,
								fragment, subrange, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							linkAndPropagate(
								triggererInstrumentationTaskId,
								fragment, task, accessStructures,
								subrange.intersect(fragment->_range), next,
								/* Do not propagate topmost property */ false,
								cpuDependencyData
							);
							
							return true;
						}
					);
				}
				
				return true;
			}
		);
	}
	
	
	static inline void replaceMatchingInBottomMapLinkAndPropagate(
		Task *task,  TaskDataAccesses &accessStructures,
		DataAccessRange range, bool weak,
		Task *parent, TaskDataAccesses &parentAccessStructures,
		/* inout */ CPUDependencyData &cpuDependencyData
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!parentAccessStructures.hasBeenDeleted());
		
		// The satisfiability propagation will decrease the predecessor count as needed
		if (!weak) {
			task->increasePredecessors();
		}
		
		bool local = false;
		#ifndef NDEBUG
			bool lastWasLocal = false;
			bool first = true;
		#endif
		
		// Link accesses to their corresponding predecessor
		foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRange(
			parent, parentAccessStructures,
			range, task,
			[&](DataAccess *previous, BottomMapEntry *bottomMapEntry) -> bool {
				assert(previous != nullptr);
				assert(previous->isReachable());
				assert(!previous->hasBeenDiscounted());
				
				Task *previousTask = previous->_originator;
				assert(previousTask != nullptr);
				
				if (bottomMapEntry != nullptr) {
					local = bottomMapEntry->_local;
				} else {
					// The first subaccess of a parent access
					local = false;
				}
				
				#ifndef NDEBUG
					if (!first) {
						assert((local == lastWasLocal) && "This fails with wrongly nested regions");
					}
					first = false;
					lastWasLocal = local;
				#endif
				
				TaskDataAccesses &previousAccessStructures = previousTask->getDataAccesses();
				assert(!previousAccessStructures.hasBeenDeleted());
				assert(previous->_range.fullyContainedIn(range));
				
				previous = linkAndPropagate(
					task->getInstrumentationTaskId(),
					previous, previousTask, previousAccessStructures,
					previous->_range, task,
					/* Propagate topmost property */ true,
					cpuDependencyData
				);
				
				return true;
			},
			[&](DataAccessRange missingRange) -> bool {
				assert(!parentAccessStructures._accesses.contains(missingRange));
				
				// Not part of the parent
				local = true;
				
				#ifndef NDEBUG
					if (!first) {
						assert((local == lastWasLocal) && "This fails with wrongly nested regions");
					}
					first = false;
					lastWasLocal = local;
				#endif
				
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
							/* Consider blocking */ true
						);
						
						targetAccess->readSatisfied() = true;
						targetAccess->writeSatisfied() = true;
						targetAccess->isTopmost() = true;
						Instrument::newDataAccessProperty(targetAccess->_instrumentationId, "T", "Topmost", task->getInstrumentationTaskId());
						
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
			},
			true /* Erase the entry from the bottom map */
		);
		
		// Add the entry to the bottom map
		BottomMapEntry *bottomMapEntry = new BottomMapEntry(range, task, local);
		parentAccessStructures._subaccessBottomMap.insert(*bottomMapEntry);
	}
	
	
	static inline void linkTaskAccesses(
		/* OUT */ CPUDependencyData &cpuDependencyData,
		Task *task
	) {
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
					
					dataAccess->isInBottomMap() = true;
					
#ifndef NDEBUG
					dataAccess->isReachable() = true;
#endif
					
					replaceMatchingInBottomMapLinkAndPropagate(
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
		}
	}
	
	
	static inline void finalizeFragments(
		Task *task, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		
		// Mark the fragments as completed and propagate topmost property
		accessStructures._accessFragments.processAll(
			[&](TaskDataAccesses::access_fragments_t::iterator position) -> bool {
				DataAccess *fragment = &(*position);
				assert(fragment != nullptr);
				assert(fragment->isFragment());
				assert(!fragment->hasBeenDiscounted());
				
				if (fragment->complete()) {
					return true;
				}
				
				Instrument::completedDataAccess(fragment->_instrumentationId, task->getInstrumentationTaskId());
				assert(!fragment->complete());
				fragment->complete() = true;
				
				bool becomesRemovable = fragment->isRemovable(fragment->hasForcedRemoval());
				
				// The next can only become topmost if it belongs to the inner dependency domain
				bool makesNextTopmost = becomesRemovable && (fragment->_next != nullptr) && (fragment->_next->getParent() == task);
				
				if (makesNextTopmost) {
					makeRangeTopmost(
						task->getInstrumentationTaskId(),
						fragment->_range, fragment->_next,
						cpuDependencyData
					);
				}
				
				// Update the number of non removable accesses if the fragment has become removable
				if (becomesRemovable) {
					handleAccessRemoval(
						fragment, accessStructures, task,
						task->getInstrumentationTaskId(), cpuDependencyData
					);
					assert(accessStructures._removalBlockers >= 0);
				}
				
				return true;
			}
		);
	}
	
	
	static inline void finalizeAccess(
		Task *finishedTask, DataAccess *dataAccess, DataAccessRange range,
		/* OUT */ CPUDependencyData &cpuDependencyData
	) {
		assert(finishedTask != nullptr);
		assert(dataAccess != nullptr);
		
		assert(dataAccess->_originator == finishedTask);
		assert(!range.empty());
		
		// The access may already have been released through the "release" directive
		if (dataAccess->complete()) {
			return;
		}
		assert(!dataAccess->hasBeenDiscounted());
		
		TaskDataAccesses &accessStructures = finishedTask->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		// Fragment if necessary
		if (!dataAccess->_range.fullyContainedIn(range)) {
			dataAccess = fragmentAccess(
				finishedTask->getInstrumentationTaskId(),
				dataAccess, range,
				accessStructures,
				/* Do not consider blocking */ false
			);
			assert(dataAccess != nullptr);
		}
		range = dataAccess->_range;
		
		assert(!dataAccess->hasForcedRemoval() || (dataAccess->_next == nullptr));
		
		if (dataAccess->hasSubaccesses()) {
			// Mark the fragments as completed and propagate topmost property
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
						accessStructures,
						/* Do not consider blocking */ false
					);
					assert(fragment != nullptr);
					
					Instrument::completedDataAccess(fragment->_instrumentationId, finishedTask->getInstrumentationTaskId());
					assert(!fragment->complete());
					fragment->complete() = true;
					
					bool becomesRemovable = fragment->isRemovable(fragment->hasForcedRemoval());
					
					// The next can only become topmost if it belongs to the inner dependency domain
					bool makesNextTopmost = becomesRemovable && (fragment->_next->getParent() == finishedTask);
					
					if ((fragment->_next != nullptr) && makesNextTopmost) {
						makeRangeTopmost(
							finishedTask->getInstrumentationTaskId(),
							fragment->_range, fragment->_next,
							cpuDependencyData
						);
					}
					
					// Update the number of non removable accesses if the fragment has become removable
					if (becomesRemovable) {
						handleAccessRemoval(
							fragment, accessStructures, finishedTask,
							finishedTask->getInstrumentationTaskId(), cpuDependencyData
						);
						assert(accessStructures._removalBlockers > 0);
					}
					
					return true;
				}
			);
			
			// Link bottom map subaccesses to the next of the current access and remove them from the bottom map
			if (dataAccess->_next != nullptr) {
				// This also propagates
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				DelayedOperation delayedOperation;
#else
				DelayedOperation &delayedOperation = getNewDelayedOperation(cpuDependencyData);
				delayedOperation._operationType = DelayedOperation::link_bottom_map_accesses_operation;
#endif
				
				delayedOperation._next = dataAccess->_next;
				delayedOperation._range = dataAccess->_range;
				delayedOperation._target = dataAccess->_originator;
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				linkBottomMapAccessesToNext(finishedTask->getInstrumentationTaskId(), delayedOperation, cpuDependencyData);
#endif
			}
		}
		
		// Mark it as complete
		Instrument::completedDataAccess(dataAccess->_instrumentationId, finishedTask->getInstrumentationTaskId());
		assert(!dataAccess->complete());
		dataAccess->complete() = true;
		
		bool becomesRemovable = dataAccess->isRemovable(dataAccess->hasForcedRemoval());
		Task *next = dataAccess->_next;
		
		// The next can only become topmost if it belongs to the same dependency domain
		bool makesNextTopmost = becomesRemovable && (next != nullptr) && (next->getParent() == finishedTask->getParent());
		
		if (!dataAccess->hasSubaccesses() && (next != nullptr)) {
			// This call will actually decide whether there is anything to propagate
			propagateSatisfiability(
				finishedTask->getInstrumentationTaskId(),
				dataAccess, next,
				makesNextTopmost,
				/* OUT */ cpuDependencyData
			);
		}
		
		// Propagate topmost property to next even if there are subaccesses
		if (dataAccess->hasSubaccesses() && makesNextTopmost) {
			makeRangeTopmost(finishedTask->getInstrumentationTaskId(), dataAccess->_range, next, cpuDependencyData);
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
		if (becomesRemovable) {
			handleAccessRemoval(
				dataAccess, accessStructures, finishedTask,
				finishedTask->getInstrumentationTaskId(), cpuDependencyData
			);
		}
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
	//! \param[in,out] task the task that performs the access
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
				task->getInstrumentationTaskId(), cpuDependencyData, cpu, currentThread
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
	
	
	static inline void makeLocalAccessesRemovable(
		Task *task, TaskDataAccesses &accessStructures,
		CPUDependencyData &cpuDependencyData
	) {
		assert(task != 0);
		assert(&accessStructures == &task->getDataAccesses());
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		TaskDataAccesses::subaccess_bottom_map_t &bottomMap = accessStructures._subaccessBottomMap;
		bottomMap.processAll(
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
				BottomMapEntry *bottomMapEntry = &(*position);
				assert(bottomMapEntry != nullptr);
				
				if (!bottomMapEntry->_local) {
					return true;
				}
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != task);
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				assert(!subtaskAccessStructures.hasBeenDeleted());
				
				TaskDataAccesses::accesses_t &subaccesses = subtaskAccessStructures._accesses;
				std::lock_guard<TaskDataAccesses::spinlock_t> subTaskGuard(subtaskAccessStructures._lock);
				
				subaccesses.processIntersecting(
					bottomMapEntry->_range,
					[&](TaskDataAccesses::accesses_t::iterator position2) -> bool {
						DataAccess *dataAccess = &(*position2);
						assert(dataAccess != nullptr);
						
						dataAccess = fragmentAccess(
							task->getInstrumentationTaskId(),
							dataAccess, bottomMapEntry->_range,
							subtaskAccessStructures,
							/* Consider blocking */ true
						);
						
						if (dataAccess->hasForcedRemoval()) {
							return true;
						}
						
						assert(dataAccess->_next == nullptr);
						dataAccess->hasForcedRemoval() = true;
						
						Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal", task->getInstrumentationTaskId());
						
						// Handle propagation of forced removal of accesses
						if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
							activateForcedRemovalOfBottomMapAccesses(
								subtask, subtaskAccessStructures,
								dataAccess->_range,
								cpuDependencyData, task->getInstrumentationTaskId()
							);
						}
						
						// Update the number of non removable accesses of the task
						if (dataAccess->isRemovable(true)) {
							assert(!dataAccess->isRemovable(false));
							
							handleAccessRemoval(
								dataAccess, subtaskAccessStructures, subtask,
								task->getInstrumentationTaskId(), cpuDependencyData
							);
						}
						
						return true;
					}
				);
				
				return true;
			}
		);
	}
	
	
	
	static inline void releaseAccessRange(Task *task, DataAccessRange range, __attribute__((unused)) DataAccessType accessType, __attribute__((unused)) bool weak)
	{
		assert(task != 0);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		
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
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.processIntersecting(
				range,
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(dataAccess->_type == accessType);
					assert(dataAccess->_weak == weak);
					
					finalizeAccess(
						task, dataAccess, range,
						/* OUT */ cpuDependencyData
					);
					
					return true;
				}
			);
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
			task->getInstrumentationTaskId(), cpuDependencyData, cpu, currentThread
		);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
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
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					
					finalizeAccess(
						task, dataAccess, dataAccess->_range,
						/* OUT */ cpuDependencyData
					);
					
					return true;
				}
			);
			
			// Mark local accesses in the bottom map as removable
			makeLocalAccessesRemovable(task, accessStructures, cpuDependencyData);
		}
		
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
			task->getInstrumentationTaskId(), cpuDependencyData, cpu, currentThread
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
			
			finalizeFragments(task, accessStructures, cpuDependencyData);
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
			task->getInstrumentationTaskId(), cpuDependencyData, cpu, currentThread
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
					
					if (dataAccess->isRemovable(dataAccess->hasForcedRemoval())) {
						// Already discounted
					} else {
						assert(accessStructures._removalBlockers > 0);
						accessStructures._removalBlockers--;
						assert(accessStructures._removalBlockers >= 0);
					}
					
					Instrument::removedDataAccess(dataAccess->_instrumentationId, task->getInstrumentationTaskId());
					accessStructures._accessFragments.erase(dataAccess);
					delete dataAccess;
					
					return true;
				}
			);
			accessStructures._accessFragments.clear();
		}
	}
	
	
	static void handleTaskRemoval(__attribute__((unused)) Task *task)
	{
	}
	
};


#endif // DATA_ACCESS_REGISTRATION_HPP
