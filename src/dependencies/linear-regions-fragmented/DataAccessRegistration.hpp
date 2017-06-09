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
#include "hardware/places/ComputePlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include "TaskDataAccessesImplementation.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentComputePlaceId.hpp>
#include <InstrumentLogMessage.hpp>
#include <InstrumentTaskId.hpp>


class DataAccessRegistration {
public:
	typedef CPUDependencyData::removable_task_list_t removable_task_list_t;
	
	
private:
	typedef CPUDependencyData::DelayedOperation DelayedOperation;
	typedef CPUDependencyData::PropagationBits PropagationBits;
	
	
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
	
	
	static inline void upgradeAccess(
		DataAccess *dataAccess, DataAccessType accessType, bool weak
	) {
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
				false
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
							Instrument::fragmentedDataAccess(originalDataAccess->_instrumentationId, fragment->_range);
						if (fragment->isTopmost()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "T", "Topmost");
						}
						if (fragment->hasPropagatedReadSatisfiability()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropR", "Propagated Read Satisfiability");
						}
						if (fragment->hasPropagatedWriteSatisfiability()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropW", "Propagated Write Satisfiability");
						}
					} else {
						Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range);
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
							Instrument::fragmentedDataAccess(originalDataAccess->_instrumentationId, fragment->_range);
						if (fragment->isTopmost()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "T", "Topmost");
						}
						if (fragment->hasPropagatedReadSatisfiability()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropR", "Propagated Read Satisfiability");
						}
						if (fragment->hasPropagatedWriteSatisfiability()) {
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropW", "Propagated Write Satisfiability");
						}
					} else {
						Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range);
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
		/* INOUT */ CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace
	) {
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : hpDependencyData._satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			ComputePlace *idleComputePlace = Scheduler::addReadyTask(satisfiedOriginator, computePlace, SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT);
			
			if (idleComputePlace != nullptr) {
				ThreadManager::resumeIdle((CPU *) idleComputePlace);
			}
		}
		
		hpDependencyData._satisfiedOriginators.clear();
	}
	
	
	static inline DelayedOperation &getNewDelayedOperation(/* OUT */ CPUDependencyData &hpDependencyData)
	{
		hpDependencyData._delayedOperations.emplace_back();
		return hpDependencyData._delayedOperations.back();
	}
	
	
	static inline void handleAccessRemoval(
		DataAccess *targetAccess, TaskDataAccesses &targetTaskAccessStructures, Task *targetTask,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		assert(targetTaskAccessStructures._removalBlockers > 0);
		targetTaskAccessStructures._removalBlockers--;
		targetAccess->markAsDiscounted();
		Instrument::dataAccessBecomesRemovable(targetAccess->_instrumentationId);
		
		if (targetAccess->_next != nullptr) {
			Instrument::unlinkedDataAccesses(
				targetAccess->_instrumentationId,
				targetAccess->_next->getInstrumentationTaskId(),
				/* direct */ true
			);
		}
		
		if (targetTaskAccessStructures._removalBlockers == 0) {
			if (targetTask->decreaseRemovalBlockingCount()) {
				hpDependencyData._removableTasks.push_back(targetTask);
			}
		}
		
		assert(targetAccess->hasForcedRemoval() || !targetAccess->isInBottomMap());
	}
	
	
	static inline PropagationBits calculatePropagationBits(
		DataAccess *dataAccess, bool wasRemovable
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		
		Task *next = dataAccess->_next;
		
		PropagationBits result;
		
		// Only propagate when there is a next to propagate to
		if (next != nullptr) {
			result._read =
				dataAccess->readSatisfied()
				&& !dataAccess->hasPropagatedReadSatisfiability()
				&& (dataAccess->complete() || (dataAccess->_type == READ_ACCESS_TYPE) || dataAccess->isFragment());
			result._write =
				dataAccess->writeSatisfied()
				&& !dataAccess->hasPropagatedWriteSatisfiability()
				&& (dataAccess->complete() || dataAccess->isFragment());
			result._concurrent =
				!dataAccess->hasPropagatedConcurrentSatisfiability()
				&& ( result._write  ||  (dataAccess->concurrentSatisfied() && (dataAccess->_type == CONCURRENT_ACCESS_TYPE)) );
		}
		
		result._becomesRemovable =
			!wasRemovable
			&& dataAccess->isRemovable(dataAccess->hasForcedRemoval(), result._read, result._write);
		
		// The next can become topmost only when this one becomes removable
		if (result._becomesRemovable) {
			if (next != nullptr) {
				assert(dataAccess->_originator != nullptr);
				
				// Find out the task that would be the parent of the next in case it became the topmost of the domain
				Task *domainParent;
				if (dataAccess->isFragment()) {
					domainParent = dataAccess->_originator;
				} else {
					domainParent = dataAccess->_originator->getParent();
				}
				assert(domainParent != nullptr);
				
				result._makesNextTopmost = (next->getParent() == domainParent);
			}
		}
		
		return result;
	}
	
	
	static inline void updatePropagation(
		DataAccess *dataAccess, PropagationBits const &propagationBits
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		
		if (propagationBits._read) {
			assert(!dataAccess->hasPropagatedReadSatisfiability());
			dataAccess->hasPropagatedReadSatisfiability() = true;
			Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropR", "Propagated Read Satisfiability");
		}
		
		if (propagationBits._write) {
			assert(!dataAccess->hasPropagatedWriteSatisfiability());
			dataAccess->hasPropagatedWriteSatisfiability() = true;
			Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropW", "Propagated Write Satisfiability");
		}
		
		if (propagationBits._concurrent) {
			assert(!dataAccess->hasPropagatedConcurrentSatisfiability());
			dataAccess->hasPropagatedConcurrentSatisfiability() = true;
			Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropC", "Propagated Concurrent Satisfiability");
		}
		
#ifndef NDEBUG
		if (propagationBits._makesNextTopmost) {
			assert(!dataAccess->hasPropagatedTopmostProperty());
			dataAccess->hasPropagatedTopmostProperty() = true;
			Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropT", "Propagated Topmost Property");
		}
#endif
	}
	
	
	static inline PropagationBits calculateAndUpdatePropagationBits(
		DataAccess *dataAccess, bool wasRemovable
	) {
		PropagationBits result = calculatePropagationBits(dataAccess, wasRemovable);
		updatePropagation(dataAccess, result);
		return result;
	}
	
	
	static inline PropagationBits calculatePropagationMask(DataAccess *dataAccess)
	{
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		
		PropagationBits result;
		result._read = dataAccess->hasPropagatedReadSatisfiability();
		result._write = dataAccess->hasPropagatedWriteSatisfiability();
		result._concurrent = dataAccess->hasPropagatedConcurrentSatisfiability();
		return result;
	}
	
	
	static inline PropagationBits applyPropagationBits(
		DataAccess *dataAccess, PropagationBits const &propagationBits
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		assert(propagationBits.propagates());
		
		bool wasRemovable = dataAccess->isRemovable(dataAccess->hasForcedRemoval());
		
		// Updates the state bits according to propagationBits
		if (propagationBits._read) {
			assert(!dataAccess->readSatisfied());
			dataAccess->readSatisfied() = true;
		}
		if (propagationBits._write) {
			assert(!dataAccess->writeSatisfied());
			dataAccess->writeSatisfied() = true;
		}
		if (propagationBits._concurrent) {
			assert(!dataAccess->concurrentSatisfied());
			dataAccess->concurrentSatisfied() = true;
		}
		
		if (propagationBits._makesNextTopmost) {
			assert(!dataAccess->isTopmost());
			dataAccess->isTopmost() = true;
			Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "T", "Topmost");
		}
		
		// NOTE: _becomesRemovable refers only to the access that produces the propagation.
		// Hence it is ignored here
		
		// Calculates the new propagation and returns it
		return calculateAndUpdatePropagationBits(dataAccess, wasRemovable);
	}
	
	
	static inline void propagateSatisfiabilityToFragments(
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		PropagationBits const &propagationBits = delayedOperation._propagationBits;
		Task *targetTask = delayedOperation._target;
		DataAccessRange range = delayedOperation._range;
		
		assert(propagationBits.propagates());
		assert(targetTask != nullptr);
		assert(!range.empty());
		
		TaskDataAccesses &targetTaskAccessStructures = targetTask->getDataAccesses();
		assert(!targetTaskAccessStructures.hasBeenDeleted());
		assert(targetTaskAccessStructures._lock.isLockedByThisThread());
		
		// NOTE: An access is discounted before traversing the fragments, so by the time we reach this point, the counter could be 0
		
		targetTaskAccessStructures._accessFragments.processIntersecting(
			range,
			[&](TaskDataAccesses::access_fragments_t::iterator position) -> bool {
				DataAccess *targetFragment = &(*position);
				assert(targetFragment != nullptr);
				assert(targetFragment->isFragment());
				assert(targetFragment->isReachable());
				assert(targetFragment->_originator == targetTask);
				assert(!targetFragment->hasBeenDiscounted());
				
				// Fragment if necessary
				targetFragment = fragmentAccess(
					targetFragment, range, targetTaskAccessStructures,
					/* Do not affect originator blocking counter */ false
				);
				assert(targetFragment != nullptr);
				assert(targetFragment->_range.fullyContainedIn(range));
				
				PropagationBits nextPropagation = applyPropagationBits(targetFragment, propagationBits);
				
				Instrument::dataAccessBecomesSatisfied(
					targetFragment->_instrumentationId,
					propagationBits._read, propagationBits._write, /* propagationBits._concurrent, */ false,
					targetTask->getInstrumentationTaskId()
				);
				
				Task *nextTask = targetFragment->_next;
				
				assert((nextTask != nullptr) || targetFragment->isInBottomMap());
				
				// Update the number of non removable accesses of the task
				if (nextPropagation._becomesRemovable) {
					handleAccessRemoval(targetFragment, targetTaskAccessStructures, targetTask, hpDependencyData);
				}
				
				if (nextTask == nullptr) {
					// Nothing else to propagate
					return true;
				}
				
				// Continue to next iteration if there is nothing to propagate
				if (!nextPropagation.propagates()) {
					return true;
				}
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				DelayedOperation nextOperation;
#else
				DelayedOperation &nextOperation = getNewDelayedOperation(hpDependencyData);
				nextOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
#endif
				nextOperation._propagationBits = nextPropagation;
				nextOperation._range = targetFragment->_range;
				nextOperation._target = nextTask;
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				TaskDataAccesses &nextTaskAccessStructures = nextTask->getDataAccesses();
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(nextTaskAccessStructures._lock);
				
				propagateSatisfiabilityPlain(nextOperation, hpDependencyData);
#endif
				
				return true;
			}
		);
	}
	
	
	static inline void propagateSatisfiabilityPlain(
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		PropagationBits const &propagationBits = delayedOperation._propagationBits;
		Task *targetTask = delayedOperation._target;
		DataAccessRange range = delayedOperation._range;
		
		assert(propagationBits.propagates());
		assert(targetTask != nullptr);
		assert(!range.empty());
		
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
				
				// Fragment if necessary
				targetAccess = fragmentAccess(
					targetAccess, range, targetTaskAccessStructures,
					/* Affect originator blocking counter */ true
				);
				assert(targetAccess != nullptr);
				assert(targetAccess->_range.fullyContainedIn(range));
				
				bool wasSatisfied = targetAccess->satisfied();
				
				PropagationBits nextPropagation = applyPropagationBits(targetAccess, propagationBits);
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->_instrumentationId,
					propagationBits._read, propagationBits._write, /* propagationBits._concurrent, */ false,
					targetTask->getInstrumentationTaskId()
				);
				
				
				// If the target access becomes satisfied decrease the predecessor count of the task
				// If it becomes 0 then add it to the list of satisfied originators
				if (!targetAccess->_weak && !wasSatisfied && targetAccess->satisfied()) {
					if (targetTask->decreasePredecessors()) {
						hpDependencyData._satisfiedOriginators.push_back(targetTask);
					}
				}
				
				if (targetAccess->hasSubaccesses()) {
					// Propagate to fragments
					
					// NOTE: The call to applyPropagationBits marks the access as having propagated.
					// This is correct, since the actual propagation will be partially made from:
					// 	the fragments for the accesses that have finished
					// 	the access for accesses that have not finished (but that can propagate)
					
					// Only propagate to fragments if there is satisfiability to propagate.
					// The topmost property is internal to the inner dependency domain.
					// Otherwise we may end up accessing a fragment that has already been
					// discounted.
					if (propagationBits.propagatesSatisfiability()) {
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						DelayedOperation nextDelayedOperation;
#else
						DelayedOperation &nextDelayedOperation = getNewDelayedOperation(hpDependencyData);
						nextDelayedOperation._operationType = DelayedOperation::propagate_satisfiability_to_fragments_operation;
#endif
						nextDelayedOperation._propagationBits = propagationBits;
						nextDelayedOperation._propagationBits._makesNextTopmost = false;
						
						nextDelayedOperation._range = targetAccess->_range;
						nextDelayedOperation._target = targetTask;
						
#if NO_DEPENDENCY_DELAYED_OPERATIONS
						propagateSatisfiabilityToFragments(nextDelayedOperation, hpDependencyData);
#endif
					}
				}
				
				// Update the number of non removable accesses of the task
				if (nextPropagation._becomesRemovable) {
					handleAccessRemoval(targetAccess, targetTaskAccessStructures, targetTask, hpDependencyData);
				}
				
				// Continue to next iteration if there is nothing to propagate
				if (!nextPropagation.propagates()) {
					return true;
				}
				
				Task *nextTask = targetAccess->_next;
				assert(nextTask != nullptr);
				
				if (targetAccess->hasSubaccesses() && targetAccess->complete()) {
					// The regular propagation happens through the fragments but we may still need to propagate the topmost property
					if (nextPropagation._makesNextTopmost) {
						nextPropagation = PropagationBits();
						nextPropagation._makesNextTopmost = true;
					} else {
						return true;
					}
				}
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				DelayedOperation nextOperation;
#else
				DelayedOperation &nextOperation = getNewDelayedOperation(hpDependencyData);
				nextOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
#endif
				nextOperation._propagationBits = nextPropagation;
				nextOperation._range = targetAccess->_range;
				nextOperation._target = nextTask;
				
#if NO_DEPENDENCY_DELAYED_OPERATIONS
				TaskDataAccesses &nextTaskAccessStructures = nextTask->getDataAccesses();
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(nextTaskAccessStructures._lock);
				
				propagateSatisfiabilityPlain(nextOperation, hpDependencyData);
#endif
				
				return true;
			}
		);
	}
	
	
	static inline void activateForcedRemovalOfBottomMapAccesses(
		Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range,
		/* OUT */ CPUDependencyData &hpDependencyData
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
								dataAccess, subrange, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(dataAccess->_next == nullptr);
							dataAccess->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal");
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, hpDependencyData);
							}
							
							if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(dataAccess, subtaskAccessStructures, subtask, hpDependencyData);
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
								fragment, subrange, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(fragment->_next == nullptr);
							fragment->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "F", "Forced Removal");
							
							if (!fragment->isRemovable(false) && fragment->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(fragment, accessStructures, task, hpDependencyData);
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
		/* OUT */ CPUDependencyData &hpDependencyData
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
								dataAccess, bottomMapEntry->_range, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(dataAccess->_next == nullptr);
							dataAccess->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal");
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->_range, hpDependencyData);
							}
							
							if (!dataAccess->isRemovable(false) && dataAccess->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(dataAccess, subtaskAccessStructures, subtask, hpDependencyData);
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
								fragment, bottomMapEntry->_range, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							assert(fragment->_next == nullptr);
							fragment->hasForcedRemoval() = true;
							Instrument::newDataAccessProperty(fragment->_instrumentationId, "F", "Forced Removal");
							
							if (!fragment->isRemovable(false) && fragment->isRemovable(true)) {
								// The access has become removable
								handleAccessRemoval(fragment, accessStructures, task, hpDependencyData);
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
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		switch (delayedOperation._operationType) {
			case DelayedOperation::link_bottom_map_accesses_operation:
				linkBottomMapAccessesToNext(delayedOperation, hpDependencyData);
				break;
			case DelayedOperation::propagate_satisfiability_to_fragments_operation:
				propagateSatisfiabilityToFragments(delayedOperation, hpDependencyData);
				break;
			case DelayedOperation::propagate_satisfiability_plain_operation:
				propagateSatisfiabilityPlain(delayedOperation, hpDependencyData);
				break;
		}
	}
	
	
	static inline void processDelayedOperations(
		/* INOUT */ CPUDependencyData &hpDependencyData
	) {
		Task *lastLocked = nullptr;
		
		while (!hpDependencyData._delayedOperations.empty()) {
			DelayedOperation const &delayedOperation = hpDependencyData._delayedOperations.front();
			
			assert(delayedOperation._target != nullptr);
			if (delayedOperation._target != lastLocked) {
				if (lastLocked != nullptr) {
					lastLocked->getDataAccesses()._lock.unlock();
				}
				lastLocked = delayedOperation._target;
				lastLocked->getDataAccesses()._lock.lock();
			}
			
			processDelayedOperation(delayedOperation, hpDependencyData);
			
			hpDependencyData._delayedOperations.pop_front();
		}
		
		if (lastLocked != nullptr) {
			lastLocked->getDataAccesses()._lock.unlock();
		}
	}
	
	
	static void processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(
		CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace
	) {
		assert(computePlace != nullptr);
		
#if NO_DEPENDENCY_DELAYED_OPERATIONS
#else
		processDelayedOperations(hpDependencyData);
#endif
		
		processSatisfiedOriginators(hpDependencyData, computePlace);
		assert(hpDependencyData._satisfiedOriginators.empty());
		
		handleRemovableTasks(hpDependencyData._removableTasks, computePlace);
	}
	
	
	static inline DataAccess *createInitialFragment(
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
			Instrument::createdDataSubaccessFragment(dataAccess->_instrumentationId);
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
		fragment->concurrentSatisfied() = dataAccess->concurrentSatisfied();
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
		DataAccessRange range,
		MatchingProcessorType matchingProcessor, MissingProcessorType missingProcessor,
		bool removeBottomMapEntry
	) {
		assert(parent != nullptr);
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
							superaccessPosition, parentAccessStructures,
							missingRange, !removeBottomMapEntry, /* Out */ bottomMapEntry
						);
						assert(previous != nullptr);
						
						previous->isTopmost() = true;
						Instrument::newDataAccessProperty(previous->_instrumentationId, "T", "Topmost");
						
						assert(previous->isFragment());
						
						previous = fragmentAccess(
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
	
	
	static inline void propagate(
		PropagationBits const &propagationBits,
		DataAccessRange range, Task *next,
		/* inout */ CPUDependencyData &hpDependencyData
	) {
		assert(propagationBits.propagates());
		assert(!range.empty());
		assert(next != nullptr);
		
#if NO_DEPENDENCY_DELAYED_OPERATIONS
		DelayedOperation nextOperation;
#else
		DelayedOperation &nextOperation = getNewDelayedOperation(hpDependencyData);
		nextOperation._operationType = DelayedOperation::propagate_satisfiability_plain_operation;
#endif
		nextOperation._propagationBits = propagationBits;
		nextOperation._range = range;
		nextOperation._target = next;
		
#if NO_DEPENDENCY_DELAYED_OPERATIONS
		TaskDataAccesses &nextTaskAccessStructures = next->getDataAccesses();
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(nextTaskAccessStructures._lock);
		
		propagateSatisfiabilityPlain(nextOperation, hpDependencyData);
#endif
	}
	
	
	static inline DataAccess *linkAndPropagate(
		DataAccess *dataAccess, Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range, Task *next,
		/* inout */ CPUDependencyData &hpDependencyData
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(dataAccess->isInBottomMap());
		assert(!dataAccess->hasBeenDiscounted());
		assert(dataAccess->_range.fullyContainedIn(range));
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(next != nullptr);
		
		dataAccess = fragmentAccess(dataAccess, range, accessStructures, /* Consider blocking */ true);
		assert(dataAccess != nullptr);
		assert(dataAccess->_next == nullptr);
		
		assert(!dataAccess->hasForcedRemoval());
		assert(!dataAccess->isRemovable(false));
		
		// Link the dataAccess
		dataAccess->_next = next;
		dataAccess->isInBottomMap() = false;
		
		Instrument::linkedDataAccesses(
			dataAccess->_instrumentationId, next->getInstrumentationTaskId(),
			dataAccess->_range,
			true, false
		);
		
		// Calculate the propagation mask before performing any other change
		PropagationBits propagationMask = calculatePropagationMask(dataAccess);
		
		// Update the propagation bits and calculate the propagation.
		// NOTE: Even if the propagation is performed through the fragments, the propagation bits must be updated.
		PropagationBits propagationBits = calculateAndUpdatePropagationBits(dataAccess, /* Was not removable */ false);
		
		if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
			// Deep-link (and propagate to) the next
			
			// This operation cannot be delayed since otherwise there could be update races
			DelayedOperation delayedOperation;
			delayedOperation._propagationBits = propagationMask;
			delayedOperation._next = next;
			delayedOperation._range = dataAccess->_range;
			delayedOperation._target = task;
			linkBottomMapAccessesToNext(delayedOperation, hpDependencyData);
			
			// The next could become topmost
			if (propagationBits._makesNextTopmost) {
				PropagationBits makeNextTopmostBits;
				makeNextTopmostBits._makesNextTopmost = true;
				propagate(makeNextTopmostBits, dataAccess->_range, next, hpDependencyData);
			}
		} else if (propagationBits.propagates()) {
			// Regular propagation
			
			assert(!dataAccess->complete() || !dataAccess->hasSubaccesses());
			propagate(propagationBits, dataAccess->_range, next, hpDependencyData);
		}
		
		// Update the number of non removable accesses of the task
		if (propagationBits._becomesRemovable) {
			handleAccessRemoval(dataAccess, accessStructures, task, hpDependencyData);
		}
		
		// Return the data access since it may have been fragmented
		return dataAccess;
	}
	
	
	static inline void linkBottomMapAccessesToNext(
		DelayedOperation const &delayedOperation,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		DataAccessRange range = delayedOperation._range;
		Task *task = delayedOperation._target;
		Task *next = delayedOperation._next;
		PropagationBits const &propagationMask = delayedOperation._propagationBits;
		
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
								subaccess, subrange, subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							
							// Avoid propagating satisfiability that has already been propagated by an ancestor
							if (propagationMask._read) {
								assert(!subaccess->hasPropagatedReadSatisfiability());
								subaccess->hasPropagatedReadSatisfiability() = true;
								Instrument::newDataAccessProperty(subaccess->_instrumentationId, "PropR", "Propagated Read Satisfiability");
							}
							if (propagationMask._write) {
								assert(!subaccess->hasPropagatedWriteSatisfiability());
								subaccess->hasPropagatedWriteSatisfiability() = true;
								Instrument::newDataAccessProperty(subaccess->_instrumentationId, "PropW", "Propagated Write Satisfiability");
							}
							if (propagationMask._concurrent) {
								assert(!subaccess->hasPropagatedConcurrentSatisfiability());
								subaccess->hasPropagatedConcurrentSatisfiability() = true;
								Instrument::newDataAccessProperty(subaccess->_instrumentationId, "PropC", "Propagated Concurrent Satisfiability");
							}
							
							linkAndPropagate(
								subaccess, subtask, subtaskAccessStructures,
								subrange.intersect(subaccess->_range), next,
								hpDependencyData
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
								fragment, subrange, accessStructures,
								/* Affect originator blocking counter */ true
							);
							
							// Avoid propagating satisfiability that has already been propagated by an ancestor
							if (propagationMask._read) {
								assert(!fragment->hasPropagatedReadSatisfiability());
								fragment->hasPropagatedReadSatisfiability() = true;
								Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropR", "Propagated Read Satisfiability");
							}
							if (propagationMask._write) {
								assert(!fragment->hasPropagatedWriteSatisfiability());
								fragment->hasPropagatedWriteSatisfiability() = true;
								Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropW", "Propagated Write Satisfiability");
							}
							if (propagationMask._concurrent) {
								assert(!fragment->hasPropagatedConcurrentSatisfiability());
								fragment->hasPropagatedConcurrentSatisfiability() = true;
								Instrument::newDataAccessProperty(fragment->_instrumentationId, "PropC", "Propagated Concurrent Satisfiability");
							}
							
							linkAndPropagate(
								fragment, task, accessStructures,
								subrange.intersect(fragment->_range), next,
								hpDependencyData
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
		/* inout */ CPUDependencyData &hpDependencyData
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
			range,
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
					previous, previousTask, previousAccessStructures,
					previous->_range, task,
					hpDependencyData
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
							targetAccess, missingRange, accessStructures,
							/* Consider blocking */ true
						);
						
						targetAccess->readSatisfied() = true;
						targetAccess->writeSatisfied() = true;
						targetAccess->concurrentSatisfied() = true;
						targetAccess->isTopmost() = true;
						Instrument::newDataAccessProperty(targetAccess->_instrumentationId, "T", "Topmost");
						
						if (!targetAccess->_weak) {
							task->decreasePredecessors();
						}
						
						Instrument::dataAccessBecomesSatisfied(
							targetAccess->_instrumentationId,
							true, true, /* true, */ false,
							task->getInstrumentationTaskId()
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
		/* OUT */ CPUDependencyData &hpDependencyData,
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
						false, false, /* false, */ false,
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
						hpDependencyData
					);
					
					// Relock to advance the iterator
					accessStructures._lock.lock();
					
					return true;
				}
			);
		}
	}
	
	
	static inline void finalizeFragment(
		DataAccess *fragment,
		Task *task, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		Instrument::completedDataAccess(fragment->_instrumentationId);
		assert(!fragment->complete());
		fragment->complete() = true;
		
		// Essentially calculate if the next becomes topmost and if the fragment becomes removable
		PropagationBits propagationBits = calculateAndUpdatePropagationBits(fragment, /* Was not removable */ false);
		assert(!propagationBits.propagatesSatisfiability());
		
		if ((fragment->_next != nullptr) && propagationBits.propagates() /* Topmost property */) {
			propagate(propagationBits, fragment->_range, fragment->_next, hpDependencyData);
		}
		
		// Update the number of non removable accesses if the fragment has become removable
		if (propagationBits._becomesRemovable) {
			handleAccessRemoval(fragment, accessStructures, task, hpDependencyData);
			assert(accessStructures._removalBlockers > 0);
		}
	}
	
	
	static inline void finalizeFragments(
		Task *task, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &hpDependencyData
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
				
				// The fragment can be already complete due to the use of the "release" directive
				if (fragment->complete()) {
					return true;
				}
				
				finalizeFragment(fragment, task, accessStructures, hpDependencyData);
				
				return true;
			}
		);
	}
	
	
	static inline void finalizeAccess(
		Task *finishedTask, DataAccess *dataAccess, DataAccessRange range,
		/* OUT */ CPUDependencyData &hpDependencyData
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
		dataAccess = fragmentAccess(dataAccess, range, accessStructures, /* Do not consider blocking */ false );
		assert(dataAccess != nullptr);
		range = dataAccess->_range;
		
		Task *next = dataAccess->_next;
		
		assert(!dataAccess->hasForcedRemoval() || (next == nullptr));
		
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
						fragment, range, accessStructures,
						/* Do not consider blocking */ false
					);
					assert(fragment != nullptr);
					
					finalizeFragment(fragment, finishedTask, accessStructures, hpDependencyData);
					
					return true;
				}
			);
			
			// Link bottom map subaccesses to the next of the current access and remove them from the bottom map
			if (next != nullptr) {
				// This also propagates
				DelayedOperation delayedOperation;
				delayedOperation._propagationBits = calculatePropagationMask(dataAccess);
				delayedOperation._next = next;
				delayedOperation._range = dataAccess->_range;
				delayedOperation._target = dataAccess->_originator;
				
				// The call to calculatePropagationMask must precede the following code
				if (dataAccess->readSatisfied() && !dataAccess->hasPropagatedReadSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->hasPropagatedReadSatisfiability() = true;
					Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropR", "Propagated Read Satisfiability");
				}
				if (dataAccess->writeSatisfied() && !dataAccess->hasPropagatedWriteSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->hasPropagatedWriteSatisfiability() = true;
					Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropW", "Propagated Write Satisfiability");
				}
				if (dataAccess->concurrentSatisfied() && !dataAccess->hasPropagatedConcurrentSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->hasPropagatedConcurrentSatisfiability() = true;
					Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "PropC", "Propagated Concurrent Satisfiability");
				}
				
				// Must be done synchronously
				linkBottomMapAccessesToNext(delayedOperation, hpDependencyData);
			}
		}
		
		// Mark it as complete
		Instrument::completedDataAccess(dataAccess->_instrumentationId);
		assert(!dataAccess->complete());
		dataAccess->complete() = true;
		
		PropagationBits propagationBits = calculateAndUpdatePropagationBits(dataAccess, /* Was not removable */ false);
		
		if (!dataAccess->hasSubaccesses() && (next != nullptr) && propagationBits.propagates()) {
			// Direct propagation
			propagate(propagationBits, dataAccess->_range, next, hpDependencyData);
		} else if (propagationBits._makesNextTopmost) {
			// Propagate only topmost property (the rest go through the fragments)
			PropagationBits makeNextTopmostBits;
			makeNextTopmostBits._makesNextTopmost = true;
			propagate(makeNextTopmostBits, dataAccess->_range, next, hpDependencyData);
		}
		
		// Handle propagation of forced removal of accesses
		if (dataAccess->hasForcedRemoval() && dataAccess->hasSubaccesses()) {
			activateForcedRemovalOfBottomMapAccesses(
				finishedTask, accessStructures, dataAccess->_range,
				hpDependencyData
			);
		}
		
		// Update the number of non removable accesses of the task
		if (propagationBits._becomesRemovable) {
			handleAccessRemoval(dataAccess, accessStructures, finishedTask, hpDependencyData);
		}
	}
	
	
	static void handleRemovableTasks(
		/* inout */ CPUDependencyData::removable_task_list_t &removableTasks,
		ComputePlace *computePlace
	) {
		for (Task *removableTask : removableTasks) {
			TaskFinalization::disposeOrUnblockTask(removableTask, computePlace);
		}
		removableTasks.clear();
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
	static inline bool registerTaskDataAccesses(
		Task *task,
		ComputePlace *computePlace
	) {
		assert(task != 0);
		assert(computePlace != nullptr);
		
		nanos_task_info *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		
		// This part creates the DataAccesses and calculates any possible upgrade
		taskInfo->register_depinfo(task, task->getArgsBlock());
		
		if (!task->getDataAccesses()._accesses.empty()) {
			// The blocking count is decreased once all the accesses become removable
			task->increaseRemovalBlockingCount();
			
			task->increasePredecessors(2);
			
			CPUDependencyData &hpDependencyData = computePlace->getDependencyData();
#ifndef NDEBUG
			{
				bool alreadyTaken = false;
				assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
			}
#endif
			
			// This part actually inserts the accesses into the dependency system
			linkTaskAccesses(hpDependencyData, task);
			
			processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(hpDependencyData, computePlace);
			
#ifndef NDEBUG
			{
				bool alreadyTaken = true;
				assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
			}
#endif
			
			return task->decreasePredecessors(2);
		} else {
			return true;
		}
	}
	
	
	static inline void makeLocalAccessesRemovable(
		__attribute((unused)) Task *task, TaskDataAccesses &accessStructures,
		CPUDependencyData &hpDependencyData
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
							dataAccess, bottomMapEntry->_range,
							subtaskAccessStructures,
							/* Consider blocking */ true
						);
						
						if (dataAccess->hasForcedRemoval()) {
							return true;
						}
						
						assert(dataAccess->_next == nullptr);
						dataAccess->hasForcedRemoval() = true;
						
						Instrument::newDataAccessProperty(dataAccess->_instrumentationId, "F", "Forced Removal");
						
						// Handle propagation of forced removal of accesses
						if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
							activateForcedRemovalOfBottomMapAccesses(
								subtask, subtaskAccessStructures,
								dataAccess->_range,
								hpDependencyData
							);
						}
						
						// Update the number of non removable accesses of the task
						if (dataAccess->isRemovable(true)) {
							assert(!dataAccess->isRemovable(false));
							
							handleAccessRemoval(dataAccess, subtaskAccessStructures, subtask, hpDependencyData);
						}
						
						return true;
					}
				);
				
				return true;
			}
		);
	}
	
	
	
	static inline void releaseAccessRange(
		Task *task, DataAccessRange range,
		__attribute__((unused)) DataAccessType accessType, __attribute__((unused)) bool weak,
		ComputePlace *computePlace
	) {
		assert(task != nullptr);
		assert(computePlace != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		
		CPUDependencyData &hpDependencyData = computePlace->getDependencyData();
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
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
					
					finalizeAccess(task, dataAccess, range, /* OUT */ hpDependencyData);
					
					return true;
				}
			);
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(hpDependencyData, computePlace);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}
	
	
	
	static inline void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace)
	{
		assert(task != nullptr);
		assert(computePlace != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		
		if (accesses.empty()) {
			return;
		}
		
		CPUDependencyData &hpDependencyData = computePlace->getDependencyData();
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					
					finalizeAccess(task, dataAccess, dataAccess->_range, /* OUT */ hpDependencyData);
					
					return true;
				}
			);
			
			// Mark local accesses in the bottom map as removable
			makeLocalAccessesRemovable(task, accessStructures, hpDependencyData);
		}
		
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(hpDependencyData, computePlace);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
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
	
	
	static void handleEnterTaskwait(Task *task, ComputePlace *computePlace)
	{
		assert(task != nullptr);
		assert(computePlace != nullptr);
		
		CPUDependencyData &hpDependencyData = computePlace->getDependencyData();
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
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
			
			activateForcedRemovalOfBottomMapAccesses(task, accessStructures, hpDependencyData);
			
			finalizeFragments(task, accessStructures, hpDependencyData);
		}
		processDelayedOperationsSatisfiedOriginatorsAndRemovableTasks(hpDependencyData, computePlace);
		
#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}
	
	
	static void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace)
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
					
					Instrument::removedDataAccess(dataAccess->_instrumentationId);
					accessStructures._accessFragments.erase(dataAccess);
					delete dataAccess;
					
					return true;
				}
			);
			accessStructures._accessFragments.clear();
		}
	}
	
	
	static void handleTaskRemoval(
		__attribute__((unused)) Task *task,
		__attribute__((unused)) ComputePlace *computePlace
	) {
	}
	
};


#endif // DATA_ACCESS_REGISTRATION_HPP
