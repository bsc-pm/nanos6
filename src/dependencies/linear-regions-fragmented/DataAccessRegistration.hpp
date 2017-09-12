/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
	
	
	static inline DataAccess *createAccess(
		Task *originator,
		DataAccessType accessType, bool weak, DataAccessRange range,
		bool fragment,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		DataAccess::status_t status = 0, Task *next = nullptr
	) {
		// Regular object duplication
		DataAccess *dataAccess = new DataAccess(
			accessType, weak, originator, range,
			fragment,
			reductionTypeAndOperatorIndex,
			Instrument::data_access_id_t(),
			status, next
		);
		
		return dataAccess;
	}
	
	
	
	
	static inline void upgradeAccess(
		DataAccess *dataAccess, DataAccessType accessType, bool weak, reduction_type_and_operator_index_t reductionTypeAndOperatorIndex
	) {
		assert(dataAccess != nullptr);
		assert(!dataAccess->hasBeenDiscounted());
		
		bool newWeak = dataAccess->isWeak() && weak;
		
		DataAccessType newDataAccessType = accessType;
		if (accessType != dataAccess->getType()) {
			FatalErrorHandler::failIf(
				(accessType == REDUCTION_ACCESS_TYPE) || (dataAccess->getType() == REDUCTION_ACCESS_TYPE),
				"Task ",
				(dataAccess->getOriginator()->getTaskInfo()->task_label != nullptr ?
					dataAccess->getOriginator()->getTaskInfo()->task_label :
					dataAccess->getOriginator()->getTaskInfo()->declaration_source
				),
				" has non-reduction accesses that overlap a reduction"
			);
			newDataAccessType = READWRITE_ACCESS_TYPE;
		} else {
			FatalErrorHandler::failIf(
				(accessType == REDUCTION_ACCESS_TYPE)
					&& (dataAccess->getReductionTypeAndOperatorIndex() != reductionTypeAndOperatorIndex),
				"Task ",
				(dataAccess->getOriginator()->getTaskInfo()->task_label != nullptr ?
					dataAccess->getOriginator()->getTaskInfo()->task_label :
					dataAccess->getOriginator()->getTaskInfo()->declaration_source
				),
				" has two overlapping reductions over different types or with different operators"
			);
		}
		
		dataAccess->upgrade(newWeak, newDataAccessType);
	}
	
	
	// NOTE: locking should be handled from the outside
	static inline DataAccess *duplicateDataAccess(
		DataAccess const &toBeDuplicated,
		TaskDataAccesses &accessStructures,
		bool updateTaskBlockingCount
	) {
		assert(toBeDuplicated.getOriginator() != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!toBeDuplicated.hasBeenDiscounted());
		
		// Regular object duplication
		DataAccess *newFragment = createAccess(
			toBeDuplicated.getOriginator(),
			toBeDuplicated.getType(), toBeDuplicated.isWeak(), toBeDuplicated.getAccessRange(),
			toBeDuplicated.isFragment(),
			toBeDuplicated.getReductionTypeAndOperatorIndex(),
			toBeDuplicated.getStatus(), toBeDuplicated.getNext()
		);
		
		if (updateTaskBlockingCount && !toBeDuplicated.isFragment() && !newFragment->isWeak() && !newFragment->satisfied()) {
			toBeDuplicated.getOriginator()->increasePredecessors();
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
		if (bottomMapEntry->getAccessRange().fullyContainedIn(range)) {
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
		assert(bottomMapEntry->getAccessRange().fullyContainedIn(range));
		
		return bottomMapEntry;
	}
	
	
	static inline DataAccess *fragmentAccess(
		DataAccess *dataAccess, DataAccessRange range,
		TaskDataAccesses &accessStructures
	) {
		assert(dataAccess != nullptr);
		// assert(accessStructures._lock.isLockedByThisThread()); // Not necessary when fragmenting an access that is not reachable
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		assert(&dataAccess->getOriginator()->getDataAccesses() == &accessStructures);
		assert(!accessStructures.hasBeenDeleted());
		assert(!dataAccess->hasBeenDiscounted());
		
		if (dataAccess->getAccessRange().fullyContainedIn(range)) {
			// Nothing to fragment
			return dataAccess;
		}
		
		// Partial overlapping of reductions is not supported at this time
		assert(dataAccess->getType() != REDUCTION_ACCESS_TYPE);
		
		if (dataAccess->isFragment()) {
			TaskDataAccesses::access_fragments_t::iterator position =
				accessStructures._accessFragments.iterator_to(*dataAccess);
			position = accessStructures._accessFragments.fragmentByIntersection(
				position, range,
				false,
				[&](DataAccess const &toBeDuplicated) -> DataAccess * {
					return duplicateDataAccess(toBeDuplicated, accessStructures, /* Count Blocking */ true);
				},
				[&](DataAccess *fragment, DataAccess *originalDataAccess) {
					if (fragment != originalDataAccess) {
						fragment->setUpNewFragment(originalDataAccess->getInstrumentationId());
					}
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->getAccessRange().fullyContainedIn(range));
		} else {
			TaskDataAccesses::accesses_t::iterator position =
				accessStructures._accesses.iterator_to(*dataAccess);
			position = accessStructures._accesses.fragmentByIntersection(
				position, range,
				false,
				[&](DataAccess const &toBeDuplicated) -> DataAccess * {
					return duplicateDataAccess(toBeDuplicated, accessStructures, /* Count Blocking */ true);
				},
				[&](DataAccess *fragment, DataAccess *originalDataAccess) {
					if (fragment != originalDataAccess) {
						fragment->setUpNewFragment(originalDataAccess->getInstrumentationId());
					}
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->getAccessRange().fullyContainedIn(range));
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
		
		if (targetAccess->getNext() != nullptr) {
			Instrument::unlinkedDataAccesses(
				targetAccess->getInstrumentationId(),
				targetAccess->getNext()->getInstrumentationTaskId(),
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
		
		Task *next = dataAccess->getNext();
		
		PropagationBits result;
		
		// Only propagate when there is a next to propagate to
		if (next != nullptr) {
			result._read =
				dataAccess->readSatisfied()
				&& !dataAccess->hasPropagatedReadSatisfiability()
				&& (dataAccess->complete() || (dataAccess->getType() == READ_ACCESS_TYPE) || dataAccess->isFragment());
			result._write =
				dataAccess->writeSatisfied()
				&& !dataAccess->hasPropagatedWriteSatisfiability()
				&& (dataAccess->complete() || dataAccess->isFragment());
			result._concurrent =
				!dataAccess->hasPropagatedConcurrentSatisfiability()
				&& ( result._write  ||  (dataAccess->concurrentSatisfied() && (dataAccess->getType() == CONCURRENT_ACCESS_TYPE)) );
			
			if (
				!dataAccess->hasPropagatedAnyReductionSatisfiability()
				&& dataAccess->writeSatisfied()
				&& (dataAccess->complete() || dataAccess->isFragment())
			) {
				result._reductionTypeAndOperatorIndex = any_reduction_type_and_operator;
			} else if (
				!dataAccess->hasPropagatedMatchingReductionSatisfiability()
				&& (dataAccess->matchingReductionSatisfied() || dataAccess->anyReductionSatisfied())
				&& (dataAccess->getType() == REDUCTION_ACCESS_TYPE)
			) {
				result._reductionTypeAndOperatorIndex = dataAccess->getReductionTypeAndOperatorIndex();
			} else {
				result._reductionTypeAndOperatorIndex = no_reduction_type_and_operator;
			}
		}
		
		result._becomesRemovable =
			!wasRemovable
			&& dataAccess->isRemovable(dataAccess->hasForcedRemoval(), result._read, result._write);
		
		// The next can become topmost only when this one becomes removable
		if (result._becomesRemovable) {
			if (next != nullptr) {
				assert(dataAccess->getOriginator() != nullptr);
				
				// Find out the task that would be the parent of the next in case it became the topmost of the domain
				Task *domainParent;
				if (dataAccess->isFragment()) {
					domainParent = dataAccess->getOriginator();
				} else {
					domainParent = dataAccess->getOriginator()->getParent();
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
			dataAccess->setPropagatedReadSatisfiability();
		}
		
		if (propagationBits._write) {
			dataAccess->setPropagatedWriteSatisfiability();
		}
		
		if (propagationBits._concurrent) {
			dataAccess->setPropagatedConcurrentSatisfiability();
		}
		
		if (propagationBits._reductionTypeAndOperatorIndex == any_reduction_type_and_operator) {
			dataAccess->setPropagatedAnyReductionSatisfiability();
			
			if ((dataAccess->getType() == REDUCTION_ACCESS_TYPE) && !dataAccess->hasPropagatedMatchingReductionSatisfiability()) {
				dataAccess->setPropagatedMatchingReductionSatisfiability();
			}
		} else if (propagationBits._reductionTypeAndOperatorIndex == no_reduction_type_and_operator) {
			// Nothing to do
		} else if (propagationBits._reductionTypeAndOperatorIndex == dataAccess->getReductionTypeAndOperatorIndex()) {
			dataAccess->setPropagatedMatchingReductionSatisfiability();
		} else {
			assert("Propagating a mismatched reduction type or operator" == nullptr);
		}
		
#ifndef NDEBUG
		if (propagationBits._makesNextTopmost) {
			dataAccess->setPropagatedTopmostProperty();
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
		
		if (dataAccess->hasPropagatedAnyReductionSatisfiability()) {
			result._reductionTypeAndOperatorIndex = any_reduction_type_and_operator;
		} else if (dataAccess->hasPropagatedMatchingReductionSatisfiability()) {
			result._reductionTypeAndOperatorIndex = dataAccess->getReductionTypeAndOperatorIndex();
		} else {
			result._reductionTypeAndOperatorIndex = no_reduction_type_and_operator;
		}
		
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
			dataAccess->setReadSatisfied();
		}
		if (propagationBits._write) {
			dataAccess->setWriteSatisfied();
		}
		if (propagationBits._concurrent) {
			dataAccess->setConcurrentSatisfied();
		}
		
		if (propagationBits._reductionTypeAndOperatorIndex == any_reduction_type_and_operator) {
			dataAccess->setAnyReductionSatisfied();
		} else if (
			(propagationBits._reductionTypeAndOperatorIndex != no_reduction_type_and_operator)
			&& (propagationBits._reductionTypeAndOperatorIndex == dataAccess->getReductionTypeAndOperatorIndex())
		) {
			dataAccess->setMatchingReductionSatisfied();
		}
		
		if (propagationBits._makesNextTopmost) {
			dataAccess->setTopmost();
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
				assert(targetFragment->getOriginator() == targetTask);
				assert(!targetFragment->hasBeenDiscounted());
				
				// Fragment if necessary
				targetFragment = fragmentAccess(targetFragment, range, targetTaskAccessStructures);
				assert(targetFragment != nullptr);
				assert(targetFragment->getAccessRange().fullyContainedIn(range));
				
				PropagationBits nextPropagation = applyPropagationBits(targetFragment, propagationBits);
				
				Instrument::dataAccessBecomesSatisfied(
					targetFragment->getInstrumentationId(),
					propagationBits._read, propagationBits._write, /* propagationBits._concurrent, */ false,
					targetTask->getInstrumentationTaskId()
				);
				
				Task *nextTask = targetFragment->getNext();
				
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
				nextOperation._range = targetFragment->getAccessRange();
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
				assert(targetAccess->getOriginator() == targetTask);
				assert(!targetAccess->hasBeenDiscounted());
				
				// Fragment if necessary
				targetAccess = fragmentAccess(targetAccess, range, targetTaskAccessStructures);
				assert(targetAccess != nullptr);
				assert(targetAccess->getAccessRange().fullyContainedIn(range));
				
				bool wasSatisfied = targetAccess->satisfied();
				
				PropagationBits nextPropagation = applyPropagationBits(targetAccess, propagationBits);
				
				Instrument::dataAccessBecomesSatisfied(
					targetAccess->getInstrumentationId(),
					propagationBits._read, propagationBits._write, /* propagationBits._concurrent, */ false,
					targetTask->getInstrumentationTaskId()
				);
				
				
				// If the target access becomes satisfied decrease the predecessor count of the task
				// If it becomes 0 then add it to the list of satisfied originators
				if (!targetAccess->isWeak() && !wasSatisfied && targetAccess->satisfied()) {
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
						
						nextDelayedOperation._range = targetAccess->getAccessRange();
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
				
				Task *nextTask = targetAccess->getNext();
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
				nextOperation._range = targetAccess->getAccessRange();
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
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->getAccessRange());
				
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
							
							assert(dataAccess->getNext() == nullptr);
							assert(dataAccess->isInBottomMap());
							assert(!dataAccess->hasBeenDiscounted());
							
							assert(!dataAccess->hasForcedRemoval());
							
							dataAccess = fragmentAccess(dataAccess, subrange, subtaskAccessStructures);
							
							assert(dataAccess->getNext() == nullptr);
							dataAccess->forceRemoval();
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->getAccessRange(), hpDependencyData);
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
							assert(fragment->getNext() == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(fragment, subrange, accessStructures);
							
							assert(fragment->getNext() == nullptr);
							fragment->forceRemoval();
							
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
						bottomMapEntry->getAccessRange(),
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *dataAccess = &(*accessPosition);
							
							assert(dataAccess->getNext() == nullptr);
							assert(dataAccess->isInBottomMap());
							assert(!dataAccess->hasBeenDiscounted());
							
							assert(!dataAccess->hasForcedRemoval());
							
							dataAccess = fragmentAccess(dataAccess, bottomMapEntry->getAccessRange(), subtaskAccessStructures);
							
							assert(dataAccess->getNext() == nullptr);
							dataAccess->forceRemoval();
							
							if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
								activateForcedRemovalOfBottomMapAccesses(subtask, subtaskAccessStructures, dataAccess->getAccessRange(), hpDependencyData);
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
						bottomMapEntry->getAccessRange(),
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->getNext() == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							assert(!fragment->hasForcedRemoval());
							
							fragment = fragmentAccess(fragment, bottomMapEntry->getAccessRange(), accessStructures);
							
							assert(fragment->getNext() == nullptr);
							fragment->forceRemoval();
							
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
		
		assert(!accessStructures._accessFragments.contains(dataAccess->getAccessRange()));
		
		Instrument::data_access_id_t instrumentationId =
			Instrument::createdDataSubaccessFragment(dataAccess->getInstrumentationId());
		DataAccess *fragment = new DataAccess(
			dataAccess->getType(),
			dataAccess->isWeak(),
			dataAccess->getOriginator(),
			dataAccess->getAccessRange(),
			/* A fragment */ true,
			dataAccess->getReductionTypeAndOperatorIndex(),
			instrumentationId
		);
		
		fragment->inheritFragmentStatus(dataAccess);
#ifndef NDEBUG
		fragment->setReachable();
#endif
		
		assert(fragment->readSatisfied() || !fragment->writeSatisfied());
		
		accessStructures._accessFragments.insert(*fragment);
		fragment->setInBottomMap();
		dataAccess->setHasSubaccesses();
		
		if (createSubrangeBottomMapEntry) {
			bottomMapEntry = new BottomMapEntry(dataAccess->getAccessRange(), dataAccess->getOriginator(), /* Not local */ false);
			accessStructures._subaccessBottomMap.insert(*bottomMapEntry);
		} else if (subrange != dataAccess->getAccessRange()) {
			dataAccess->getAccessRange().processIntersectingFragments(
				subrange,
				[&](DataAccessRange excludedSubrange) {
					bottomMapEntry = new BottomMapEntry(excludedSubrange, dataAccess->getOriginator(), /* Not local */ false);
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
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->getAccessRange());
				
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
							
							assert(previous->getNext() == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(previous, subrange, subtaskAccessStructures);
							
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
							
							assert(previous->getNext() == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(previous, subrange, parentAccessStructures);
							
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
						assert(previous->isFragment());
						
						previous->setTopmost();
						previous = fragmentAccess(previous, missingRange, parentAccessStructures);
						
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
		assert(dataAccess->getAccessRange().fullyContainedIn(range));
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(next != nullptr);
		
		dataAccess = fragmentAccess(dataAccess, range, accessStructures);
		assert(dataAccess != nullptr);
		assert(dataAccess->getNext() == nullptr);
		
		assert(!dataAccess->hasForcedRemoval());
		assert(!dataAccess->isRemovable(false));
		
		// Link the dataAccess
		dataAccess->setNext(next);
		dataAccess->unsetInBottomMap();
		
		Instrument::linkedDataAccesses(
			dataAccess->getInstrumentationId(), next->getInstrumentationTaskId(),
			dataAccess->getAccessRange(),
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
			delayedOperation._range = dataAccess->getAccessRange();
			delayedOperation._target = task;
			linkBottomMapAccessesToNext(delayedOperation, hpDependencyData);
			
			// The next could become topmost
			if (propagationBits._makesNextTopmost) {
				PropagationBits makeNextTopmostBits;
				makeNextTopmostBits._makesNextTopmost = true;
				propagate(makeNextTopmostBits, dataAccess->getAccessRange(), next, hpDependencyData);
			}
		} else if (propagationBits.propagates()) {
			// Regular propagation
			
			assert(!dataAccess->complete() || !dataAccess->hasSubaccesses());
			propagate(propagationBits, dataAccess->getAccessRange(), next, hpDependencyData);
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
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->getAccessRange());
				
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
							assert(subaccess->getNext() == nullptr);
							assert(subaccess->isInBottomMap());
							assert(!subaccess->hasBeenDiscounted());
							
							subaccess = fragmentAccess(subaccess, subrange, subtaskAccessStructures);
							
							// Avoid propagating satisfiability that has already been propagated by an ancestor
							if (propagationMask._read) {
								subaccess->setPropagatedReadSatisfiability();
							}
							if (propagationMask._write) {
								subaccess->setPropagatedWriteSatisfiability();
							}
							if (propagationMask._concurrent) {
								subaccess->setPropagatedConcurrentSatisfiability();
							}
							if (propagationMask._reductionTypeAndOperatorIndex == any_reduction_type_and_operator) {
								subaccess->setPropagatedAnyReductionSatisfiability();
							} else if (propagationMask._reductionTypeAndOperatorIndex != no_reduction_type_and_operator) {
								assert(propagationMask._reductionTypeAndOperatorIndex == subaccess->getReductionTypeAndOperatorIndex());
								subaccess->setPropagatedMatchingReductionSatisfiability();
							}
							
							linkAndPropagate(
								subaccess, subtask, subtaskAccessStructures,
								subrange.intersect(subaccess->getAccessRange()), next,
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
							assert(fragment->getNext() == nullptr);
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(fragment, subrange, accessStructures);
							
							// Avoid propagating satisfiability that has already been propagated by an ancestor
							if (propagationMask._read) {
								fragment->setPropagatedReadSatisfiability();
							}
							if (propagationMask._write) {
								fragment->setPropagatedWriteSatisfiability();
							}
							if (propagationMask._concurrent) {
								fragment->setPropagatedConcurrentSatisfiability();
							}
							
							if (propagationMask._reductionTypeAndOperatorIndex == any_reduction_type_and_operator) {
								fragment->setPropagatedAnyReductionSatisfiability();
							} else if (propagationMask._reductionTypeAndOperatorIndex != no_reduction_type_and_operator) {
								assert(propagationMask._reductionTypeAndOperatorIndex == fragment->getReductionTypeAndOperatorIndex());
								fragment->setPropagatedMatchingReductionSatisfiability();
							}
							
							linkAndPropagate(
								fragment, task, accessStructures,
								subrange.intersect(fragment->getAccessRange()), next,
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
				
				Task *previousTask = previous->getOriginator();
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
				assert(previous->getAccessRange().fullyContainedIn(range));
				
				previous = linkAndPropagate(
					previous, previousTask, previousAccessStructures,
					previous->getAccessRange(), task,
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
						
						targetAccess = fragmentAccess(targetAccess, missingRange, accessStructures);
						
						targetAccess->setReadSatisfied();
						targetAccess->setWriteSatisfied();
						targetAccess->setConcurrentSatisfied();
						targetAccess->setAnyReductionSatisfied();
						targetAccess->setMatchingReductionSatisfied();
						targetAccess->setTopmost();
						
						if (!targetAccess->isWeak()) {
							task->decreasePredecessors();
						}
						
						Instrument::dataAccessBecomesSatisfied(
							targetAccess->getInstrumentationId(),
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
					
					dataAccess->setNewInstrumentationId(task->getInstrumentationTaskId());
					
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
					
					dataAccess->setInBottomMap();
					
#ifndef NDEBUG
					dataAccess->setReachable();
#endif
					
					replaceMatchingInBottomMapLinkAndPropagate(
						task, accessStructures,
						dataAccess->getAccessRange(), dataAccess->isWeak(),
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
		fragment->setComplete();
		
		// Essentially calculate if the next becomes topmost and if the fragment becomes removable
		PropagationBits propagationBits = calculateAndUpdatePropagationBits(fragment, /* Was not removable */ false);
		assert(!propagationBits.propagatesSatisfiability());
		
		if ((fragment->getNext() != nullptr) && propagationBits.propagates() /* Topmost property */) {
			propagate(propagationBits, fragment->getAccessRange(), fragment->getNext(), hpDependencyData);
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
		
		assert(dataAccess->getOriginator() == finishedTask);
		assert(!range.empty());
		
		// The access may already have been released through the "release" directive
		if (dataAccess->complete()) {
			return;
		}
		assert(!dataAccess->hasBeenDiscounted());
		
		TaskDataAccesses &accessStructures = finishedTask->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		// Fragment if necessary
		dataAccess = fragmentAccess(dataAccess, range, accessStructures);
		assert(dataAccess != nullptr);
		range = dataAccess->getAccessRange();
		
		Task *next = dataAccess->getNext();
		
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
					
					fragment = fragmentAccess(fragment, range, accessStructures);
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
				delayedOperation._range = dataAccess->getAccessRange();
				delayedOperation._target = dataAccess->getOriginator();
				
				// The call to calculatePropagationMask must precede the following code
				if (dataAccess->readSatisfied() && !dataAccess->hasPropagatedReadSatisfiability()) {
					dataAccess->setPropagatedReadSatisfiability();
				}
				if (dataAccess->writeSatisfied() && !dataAccess->hasPropagatedWriteSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->setPropagatedWriteSatisfiability();
				}
				if (dataAccess->concurrentSatisfied() && !dataAccess->hasPropagatedConcurrentSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->setPropagatedConcurrentSatisfiability();
				}
				if (dataAccess->anyReductionSatisfied() && ! dataAccess->hasPropagatedAnyReductionSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->setPropagatedAnyReductionSatisfiability();
				}
				if (dataAccess->matchingReductionSatisfied() && ! dataAccess->hasPropagatedMatchingReductionSatisfiability()) {
					// The actual propagation will occur through the bottom map accesses
					dataAccess->setPropagatedMatchingReductionSatisfiability();
				}
				
				// Must be done synchronously
				linkBottomMapAccessesToNext(delayedOperation, hpDependencyData);
			}
		}
		
		// Mark it as complete
		dataAccess->setComplete();
		
		PropagationBits propagationBits = calculateAndUpdatePropagationBits(dataAccess, /* Was not removable */ false);
		
		if (!dataAccess->hasSubaccesses() && (next != nullptr) && propagationBits.propagates()) {
			// Direct propagation
			propagate(propagationBits, dataAccess->getAccessRange(), next, hpDependencyData);
		} else if (propagationBits._makesNextTopmost) {
			// Propagate only topmost property (the rest go through the fragments)
			PropagationBits makeNextTopmostBits;
			makeNextTopmostBits._makesNextTopmost = true;
			propagate(makeNextTopmostBits, dataAccess->getAccessRange(), next, hpDependencyData);
		}
		
		// Handle propagation of forced removal of accesses
		if (dataAccess->hasForcedRemoval() && dataAccess->hasSubaccesses()) {
			activateForcedRemovalOfBottomMapAccesses(
				finishedTask, accessStructures, dataAccess->getAccessRange(),
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
	//! \param[in] reductionTypeAndOperatorIndex an index that identifies the type and the operation of the reduction
	static inline void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange range, reduction_type_and_operator_index_t reductionTypeAndOperatorIndex
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
				
				upgradeAccess(oldAccess, accessType, weak, reductionTypeAndOperatorIndex);
				
				return true;
			},
			[&](DataAccessRange missingRange) -> bool {
				DataAccess *newAccess = createAccess(task, accessType, weak, missingRange, false, reductionTypeAndOperatorIndex);
				
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
					bottomMapEntry->getAccessRange(),
					[&](TaskDataAccesses::accesses_t::iterator position2) -> bool {
						DataAccess *dataAccess = &(*position2);
						assert(dataAccess != nullptr);
						
						dataAccess = fragmentAccess(dataAccess, bottomMapEntry->getAccessRange(), subtaskAccessStructures);
						
						if (dataAccess->hasForcedRemoval()) {
							return true;
						}
						
						assert(dataAccess->getNext() == nullptr);
						dataAccess->forceRemoval();
						
						// Handle propagation of forced removal of accesses
						if (dataAccess->complete() && dataAccess->hasSubaccesses()) {
							activateForcedRemovalOfBottomMapAccesses(
								subtask, subtaskAccessStructures,
								dataAccess->getAccessRange(),
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
					assert(dataAccess->getType() == accessType);
					assert(dataAccess->isWeak() == weak);
					
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
					
					finalizeAccess(task, dataAccess, dataAccess->getAccessRange(), /* OUT */ hpDependencyData);
					
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
					
					if (dataAccess->hasSubaccesses()) {
						dataAccess->unsetHasSubaccesses();
					}
					
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
					
					Instrument::removedDataAccess(dataAccess->getInstrumentationId());
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
