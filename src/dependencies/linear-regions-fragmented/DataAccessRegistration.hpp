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


#pragma GCC visibility push(hidden)

class DataAccessRegistration {
public:
	typedef CPUDependencyData::removable_task_list_t removable_task_list_t;
	
	
private:
	typedef CPUDependencyData::UpdateOperation UpdateOperation;
	
	
	
	struct DataAccessStatusEffects {
		bool _isRegistered;
		bool _isSatisfied;
		bool _enforcesDependency;
		
		bool _hasNext;
		bool _propagatesReadSatisfiabilityToNext;
		bool _propagatesWriteSatisfiabilityToNext;
		bool _propagatesConcurrentSatisfiabilityToNext;
		reduction_type_and_operator_index_t _propagatesReductionSatisfiabilityToNext;
		bool _makesNextTopmost;
		
		bool _propagatesReadSatisfiabilityToFragments;
		bool _propagatesWriteSatisfiabilityToFragments;
		bool _propagatesConcurrentSatisfiabilityToFragments;
		reduction_type_and_operator_index_t _propagatesReductionSatisfiabilityToFragments;
		
		bool _activatesForcedRemovalOfBottomMapAccesses;
		bool _linksBottomMapAccessesToNextAndInhibitsPropagation;
		
		bool _isRemovable;
		
	public:
		DataAccessStatusEffects()
			: _isRegistered(false),
			_isSatisfied(false), _enforcesDependency(false),
			
			_hasNext(false),
			_propagatesReadSatisfiabilityToNext(false), _propagatesWriteSatisfiabilityToNext(false), _propagatesConcurrentSatisfiabilityToNext(false),
			_propagatesReductionSatisfiabilityToNext(no_reduction_type_and_operator),
			_makesNextTopmost(false),
			
			_propagatesReadSatisfiabilityToFragments(false), _propagatesWriteSatisfiabilityToFragments(false), _propagatesConcurrentSatisfiabilityToFragments(false),
			_propagatesReductionSatisfiabilityToFragments(no_reduction_type_and_operator),
			
			_activatesForcedRemovalOfBottomMapAccesses(false),
			_linksBottomMapAccessesToNextAndInhibitsPropagation(false),
			
			_isRemovable(false)
		{
		}
		
		DataAccessStatusEffects(DataAccess const *access)
		{
			_isRegistered = access->isRegistered();
			
			_isSatisfied = access->satisfied();
			_enforcesDependency = !access->isWeak() && !access->satisfied();
			_hasNext = access->hasNext();
			
			// Propagation to fragments
			if (access->hasSubaccesses()) {
				_propagatesReadSatisfiabilityToFragments = access->readSatisfied();
				_propagatesWriteSatisfiabilityToFragments = access->writeSatisfied();
				_propagatesConcurrentSatisfiabilityToFragments = access->concurrentSatisfied();
				if (access->anyReductionSatisfied()) {
					_propagatesReductionSatisfiabilityToFragments = any_reduction_type_and_operator;
				} else if (access->matchingReductionSatisfied()) {
					_propagatesReductionSatisfiabilityToFragments = access->getReductionTypeAndOperatorIndex();
				} else {
					_propagatesReductionSatisfiabilityToFragments = no_reduction_type_and_operator;
				}
			} else {
				_propagatesReadSatisfiabilityToFragments = false;
				_propagatesWriteSatisfiabilityToFragments = false;
				_propagatesConcurrentSatisfiabilityToFragments = false;
				_propagatesReductionSatisfiabilityToFragments = no_reduction_type_and_operator;
			}
			
			// Propagation to next
			if (_hasNext) {
				if (access->hasSubaccesses()) {
					assert(!access->isFragment());
					_propagatesReadSatisfiabilityToNext =
						access->readSatisfied() && access->canPropagateReadSatisfiability()
						&& (access->getType() == READ_ACCESS_TYPE);
					_propagatesWriteSatisfiabilityToNext = false; // Write satisfiability is propagated through the fragments
					_propagatesConcurrentSatisfiabilityToNext =
						access->canPropagateConcurrentSatisfiability()
						&& access->concurrentSatisfied() && (access->getType() == CONCURRENT_ACCESS_TYPE);
					
					if (
						!access->canPropagateAnyReductionSatisfiability()
						&& !access->canPropagateMatchingReductionSatisfiability()
					) {
						_propagatesReductionSatisfiabilityToNext = no_reduction_type_and_operator;
					} else if (
						access->canPropagateMatchingReductionSatisfiability()
						&& (access->matchingReductionSatisfied() || access->anyReductionSatisfied())
						&& (access->getType() == REDUCTION_ACCESS_TYPE)
					) {
						_propagatesReductionSatisfiabilityToNext = access->getReductionTypeAndOperatorIndex();
					} else {
						// Reduction satisfiability of non-reductions is propagated through the fragments
						_propagatesReductionSatisfiabilityToNext = no_reduction_type_and_operator;
					}
				} else if (access->isFragment()) {
					_propagatesReadSatisfiabilityToNext =
						access->canPropagateReadSatisfiability()
						&& access->readSatisfied();
					_propagatesWriteSatisfiabilityToNext = access->writeSatisfied();
					_propagatesConcurrentSatisfiabilityToNext =
						access->canPropagateConcurrentSatisfiability()
						&& access->concurrentSatisfied();
					
					if (
						access->canPropagateAnyReductionSatisfiability()
						&& access->anyReductionSatisfied()
					) {
						_propagatesReductionSatisfiabilityToNext = any_reduction_type_and_operator;
					} else if (
						access->canPropagateMatchingReductionSatisfiability()
						&& access->matchingReductionSatisfied()
					) {
						_propagatesReductionSatisfiabilityToNext = access->getReductionTypeAndOperatorIndex();
					} else {
						_propagatesReductionSatisfiabilityToNext = no_reduction_type_and_operator;
					}
				} else {
					// A regular access without subaccesses but with a next
					_propagatesReadSatisfiabilityToNext =
						access->canPropagateReadSatisfiability()
						&& access->readSatisfied()
						&& ((access->getType() == READ_ACCESS_TYPE) || access->complete());
					_propagatesWriteSatisfiabilityToNext =
						access->writeSatisfied() && access->complete();
					_propagatesConcurrentSatisfiabilityToNext =
						access->canPropagateConcurrentSatisfiability()
						&& access->concurrentSatisfied()
						&& (access->complete() || (access->getType() == CONCURRENT_ACCESS_TYPE));
					
					if (
						access->canPropagateAnyReductionSatisfiability()
						&& access->anyReductionSatisfied()
						&& access->complete()
					) {
						_propagatesReductionSatisfiabilityToNext = any_reduction_type_and_operator;
					} else if (
						access->canPropagateMatchingReductionSatisfiability()
						&& access->anyReductionSatisfied()
						&& (access->getType() == REDUCTION_ACCESS_TYPE)
					) {
						_propagatesReductionSatisfiabilityToNext = access->getReductionTypeAndOperatorIndex();
					} else if (
						access->canPropagateMatchingReductionSatisfiability()
						&& access->matchingReductionSatisfied()
						&& (access->getType() == REDUCTION_ACCESS_TYPE)
					) {
						_propagatesReductionSatisfiabilityToNext = access->getReductionTypeAndOperatorIndex();
					} else {
						_propagatesReductionSatisfiabilityToNext = no_reduction_type_and_operator;
					}
				}
			} else {
				assert(!access->hasNext());
				_propagatesReadSatisfiabilityToNext = false;
				_propagatesWriteSatisfiabilityToNext = false;
				_propagatesConcurrentSatisfiabilityToNext = false;
				_propagatesReductionSatisfiabilityToNext = no_reduction_type_and_operator;
			}
			
			_isRemovable = access->isTopmost()
				&& access->readSatisfied() && access->writeSatisfied()
				&& access->complete()
				&& ( access->hasForcedRemoval() || !access->isInBottomMap() || access->hasNext() );
			
			if (_isRemovable && access->hasNext()) {
				assert(access->getOriginator() != nullptr);
				
				// Find out the task that would be the parent of the next in case it became the topmost of the domain
				Task *domainParent;
				if (access->isFragment()) {
					domainParent = access->getOriginator();
				} else {
					domainParent = access->getOriginator()->getParent();
				}
				assert(domainParent != nullptr);
				
				_makesNextTopmost = (access->getNext()->getParent() == domainParent);
			} else {
				_makesNextTopmost = false;
			}
			
			_activatesForcedRemovalOfBottomMapAccesses = 
				!access->isFragment()
				&& access->hasForcedRemoval() && access->complete() && access->hasSubaccesses();
			
			// NOTE: Calculate inhibition from initial status
			_linksBottomMapAccessesToNextAndInhibitsPropagation =
				access->hasNext() && access->complete() && access->hasSubaccesses();
			
		}
	};
	
	
	struct BottomMapUpdateOperation {
		DataAccessRegion _region;
		
		bool _activateForcedRemovalOfBottomMapAccesses;
		bool _linkBottomMapAccessesToNext;
		
		bool _inhibitReadPropagation;
		bool _inhibitConcurrentPropagation;
		reduction_type_and_operator_index_t _inhibitReductionPropagation;
		
		Task *_next;
		
		BottomMapUpdateOperation()
			: _region(),
			_activateForcedRemovalOfBottomMapAccesses(false),
			_linkBottomMapAccessesToNext(false),
			_inhibitReadPropagation(false),
			_inhibitConcurrentPropagation(false),
			_inhibitReductionPropagation(no_reduction_type_and_operator),
			_next(nullptr)
		{
		}
		
		BottomMapUpdateOperation(DataAccessRegion const &region)
			: _region(region),
			_activateForcedRemovalOfBottomMapAccesses(false),
			_linkBottomMapAccessesToNext(false),
			_inhibitReadPropagation(false),
			_inhibitConcurrentPropagation(false),
			_inhibitReductionPropagation(no_reduction_type_and_operator),
			_next(nullptr)
		{
		}
		
		bool empty() const
		{
			return !_activateForcedRemovalOfBottomMapAccesses && !_linkBottomMapAccessesToNext;
		}
	};
	
	
	static inline void handleDataAccessStatusChanges(
		DataAccessStatusEffects const &initialStatus,
		DataAccessStatusEffects const &updatedStatus,
		DataAccess *access, TaskDataAccesses &accessStructures, Task *task,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		// Registration
		if (initialStatus._isRegistered != updatedStatus._isRegistered) {
			assert(!initialStatus._isRegistered);
			
			// Count the access
			if (!initialStatus._isRemovable) {
				accessStructures._removalBlockers++;
			}
			
			if (updatedStatus._enforcesDependency) {
				task->increasePredecessors();
			}
		}
		
		if (!updatedStatus._isRegistered) {
			return;
		}
		
		// NOTE: After this point, all actions assume the access is registered
		
		// Satisfiability
		if (initialStatus._isSatisfied != updatedStatus._isSatisfied) {
			assert(!initialStatus._isSatisfied);
			Instrument::dataAccessBecomesSatisfied(
				access->getInstrumentationId(),
				true, true, /* true, */ false,
				task->getInstrumentationTaskId()
			);
		}
		
		// Link to Next
		if (initialStatus._hasNext != updatedStatus._hasNext) {
			assert(!initialStatus._hasNext);
			Instrument::linkedDataAccesses(
				access->getInstrumentationId(), access->getNext()->getInstrumentationTaskId(),
				access->getAccessRegion(),
				true, false
			);
		}
		
		// Dependency updates
		if (initialStatus._enforcesDependency != updatedStatus._enforcesDependency) {
			if (updatedStatus._enforcesDependency) {
				// A new access that enforces a dependency.
				// Already counted as part of the registration status change.
				assert(!initialStatus._isRegistered && updatedStatus._isRegistered);
			} else {
				// The access no longer enforces a dependency (has become satisified)
				if (task->decreasePredecessors()) {
					// The task becomes ready
					hpDependencyData._satisfiedOriginators.push_back(task);
				}
			}
		}
		
		// Propagation to Next
		if (access->hasNext()) {
			UpdateOperation updateOperation(access->getNext(), access->getAccessRegion(), /* To Next */ true);
			
			if (initialStatus._propagatesReadSatisfiabilityToNext != updatedStatus._propagatesReadSatisfiabilityToNext) {
				assert(!initialStatus._propagatesReadSatisfiabilityToNext);
				updateOperation._makeReadSatisfied = true;
			}
			
			if (initialStatus._propagatesWriteSatisfiabilityToNext != updatedStatus._propagatesWriteSatisfiabilityToNext) {
				assert(!initialStatus._propagatesWriteSatisfiabilityToNext);
				updateOperation._makeWriteSatisfied = true;
			}
			
			if (initialStatus._propagatesConcurrentSatisfiabilityToNext != updatedStatus._propagatesConcurrentSatisfiabilityToNext) {
				assert(!initialStatus._propagatesConcurrentSatisfiabilityToNext);
				updateOperation._makeConcurrentSatisfied = true;
			}
			
			if (initialStatus._propagatesReductionSatisfiabilityToNext != updatedStatus._propagatesReductionSatisfiabilityToNext) {
				assert(updatedStatus._propagatesReductionSatisfiabilityToNext != no_reduction_type_and_operator);
				updateOperation._makeReductionSatisfied = updatedStatus._propagatesReductionSatisfiabilityToNext;
			}
			
			// Make Next Topmost
			if (initialStatus._makesNextTopmost != updatedStatus._makesNextTopmost) {
				assert(!initialStatus._makesNextTopmost);
				updateOperation._makeTopmost = true;
			}
			
			if (!updateOperation.empty()) {
				hpDependencyData._delayedOperations.emplace_back(updateOperation);
			}
		}
		
		// Propagation to Fragments
		if (access->hasSubaccesses()) {
			UpdateOperation updateOperation(task, access->getAccessRegion(), /* To Fragments */ false);
			
			if (initialStatus._propagatesReadSatisfiabilityToFragments != updatedStatus._propagatesReadSatisfiabilityToFragments) {
				assert(!initialStatus._propagatesReadSatisfiabilityToFragments);
				updateOperation._makeReadSatisfied = true;
			}
			
			if (initialStatus._propagatesWriteSatisfiabilityToFragments != updatedStatus._propagatesWriteSatisfiabilityToFragments) {
				assert(!initialStatus._propagatesWriteSatisfiabilityToFragments);
				updateOperation._makeWriteSatisfied = true;
			}
			
			if (initialStatus._propagatesConcurrentSatisfiabilityToFragments != updatedStatus._propagatesConcurrentSatisfiabilityToFragments) {
				assert(!initialStatus._propagatesConcurrentSatisfiabilityToFragments);
				updateOperation._makeConcurrentSatisfied = true;
			}
			
			if (initialStatus._propagatesReductionSatisfiabilityToFragments != updatedStatus._propagatesReductionSatisfiabilityToFragments) {
				assert(updatedStatus._propagatesReductionSatisfiabilityToFragments != no_reduction_type_and_operator);
				updateOperation._makeReductionSatisfied = updatedStatus._propagatesReductionSatisfiabilityToFragments;
			}
			
			if (!updateOperation.empty()) {
				hpDependencyData._delayedOperations.emplace_back(updateOperation);
			}
		}
		
		// Bottom Map Updates
		if (access->hasSubaccesses()) {
			BottomMapUpdateOperation bottomMapUpdateOperation(access->getAccessRegion());
			
			if (initialStatus._activatesForcedRemovalOfBottomMapAccesses != updatedStatus._activatesForcedRemovalOfBottomMapAccesses) {
				assert(!initialStatus._activatesForcedRemovalOfBottomMapAccesses);
				bottomMapUpdateOperation._activateForcedRemovalOfBottomMapAccesses = true;
			}
			
			if (
				initialStatus._linksBottomMapAccessesToNextAndInhibitsPropagation
				!= updatedStatus._linksBottomMapAccessesToNextAndInhibitsPropagation
			) {
				assert(!initialStatus._linksBottomMapAccessesToNextAndInhibitsPropagation);
				bottomMapUpdateOperation._linkBottomMapAccessesToNext = true;
				bottomMapUpdateOperation._next = access->getNext();
				
				bottomMapUpdateOperation._inhibitReadPropagation = (access->getType() == READ_ACCESS_TYPE);
				assert(!updatedStatus._propagatesWriteSatisfiabilityToNext);
				bottomMapUpdateOperation._inhibitConcurrentPropagation = (access->getType() == CONCURRENT_ACCESS_TYPE);
				bottomMapUpdateOperation._inhibitReductionPropagation =
					(access->getType() == REDUCTION_ACCESS_TYPE ? access->getReductionTypeAndOperatorIndex() : no_reduction_type_and_operator);
			}
			
			if (!bottomMapUpdateOperation.empty()) {
				processBottomMapUpdate(bottomMapUpdateOperation, accessStructures, task, hpDependencyData);
			}
		}
		
		// Removable
		if (initialStatus._isRemovable != updatedStatus._isRemovable) {
			assert(!initialStatus._isRemovable);
			
			assert(accessStructures._removalBlockers > 0);
			accessStructures._removalBlockers--;
			access->markAsDiscounted();
			
			if (access->getNext() != nullptr) {
				Instrument::unlinkedDataAccesses(
					access->getInstrumentationId(),
					access->getNext()->getInstrumentationTaskId(),
					/* direct */ true
				);
			}
			
			if (accessStructures._removalBlockers == 0) {
				if (task->decreaseRemovalBlockingCount()) {
					hpDependencyData._removableTasks.push_back(task);
				}
			}
		}
	}
	
	
	static inline DataAccess *createAccess(
		Task *originator,
		DataAccessType accessType, bool weak, DataAccessRegion region,
		bool fragment,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		DataAccess::status_t status = 0, Task *next = nullptr
	) {
		// Regular object duplication
		DataAccess *dataAccess = new DataAccess(
			accessType, weak, originator, region,
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
		__attribute__((unused)) TaskDataAccesses &accessStructures
	) {
		assert(toBeDuplicated.getOriginator() != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!toBeDuplicated.hasBeenDiscounted());
		
		// Regular object duplication
		DataAccess *newFragment = createAccess(
			toBeDuplicated.getOriginator(),
			toBeDuplicated.getType(), toBeDuplicated.isWeak(), toBeDuplicated.getAccessRegion(),
			toBeDuplicated.isFragment(),
			toBeDuplicated.getReductionTypeAndOperatorIndex(),
			toBeDuplicated.getStatus(), toBeDuplicated.getNext()
		);
		
		newFragment->clearRegistered();
		
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
		BottomMapEntry *bottomMapEntry, DataAccessRegion region,
		TaskDataAccesses &accessStructures
	) {
		if (bottomMapEntry->getAccessRegion().fullyContainedIn(region)) {
			// Nothing to fragment
			return bottomMapEntry;
		}
		
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		TaskDataAccesses::subaccess_bottom_map_t::iterator position =
			accessStructures._subaccessBottomMap.iterator_to(*bottomMapEntry);
		position = accessStructures._subaccessBottomMap.fragmentByIntersection(
			position, region,
			false,
			[&](BottomMapEntry const &toBeDuplicated) -> BottomMapEntry * {
				return new BottomMapEntry(DataAccessRegion(), toBeDuplicated._task, toBeDuplicated._local);
			},
			[&](__attribute__((unused)) BottomMapEntry *fragment, __attribute__((unused)) BottomMapEntry *originalBottomMapEntry) {
			}
		);
		
		bottomMapEntry = &(*position);
		assert(bottomMapEntry != nullptr);
		assert(bottomMapEntry->getAccessRegion().fullyContainedIn(region));
		
		return bottomMapEntry;
	}
	
	
	static inline void setUpNewFragment(
		DataAccess *fragment, DataAccess *originalDataAccess,
		TaskDataAccesses &accessStructures
	) {
		if (fragment != originalDataAccess) {
			CPUDependencyData hpDependencyData;
			
			DataAccessStatusEffects initialStatus(fragment);
			fragment->setUpNewFragment(originalDataAccess->getInstrumentationId());
			fragment->setRegistered();
			DataAccessStatusEffects updatedStatus(fragment);
			
			handleDataAccessStatusChanges(
				initialStatus, updatedStatus,
				fragment, accessStructures, fragment->getOriginator(),
				hpDependencyData
			);
			
			assert(hpDependencyData.empty());
		}
	}
	
	
	static inline DataAccess *fragmentAccess(
		DataAccess *dataAccess, DataAccessRegion region,
		TaskDataAccesses &accessStructures
	) {
		assert(dataAccess != nullptr);
		// assert(accessStructures._lock.isLockedByThisThread()); // Not necessary when fragmenting an access that is not reachable
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		assert(&dataAccess->getOriginator()->getDataAccesses() == &accessStructures);
		assert(!accessStructures.hasBeenDeleted());
		assert(!dataAccess->hasBeenDiscounted());
		
		if (dataAccess->getAccessRegion().fullyContainedIn(region)) {
			// Nothing to fragment
			return dataAccess;
		}
		
		// Partial overlapping of reductions is not supported at this time
		assert(dataAccess->getType() != REDUCTION_ACCESS_TYPE);
		
		if (dataAccess->isFragment()) {
			TaskDataAccesses::access_fragments_t::iterator position =
				accessStructures._accessFragments.iterator_to(*dataAccess);
			position = accessStructures._accessFragments.fragmentByIntersection(
				position, region,
				false,
				[&](DataAccess const &toBeDuplicated) -> DataAccess * {
					assert(toBeDuplicated.isRegistered());
					return duplicateDataAccess(toBeDuplicated, accessStructures);
				},
				[&](DataAccess *fragment, DataAccess *originalDataAccess) {
					setUpNewFragment(fragment, originalDataAccess, accessStructures);
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->getAccessRegion().fullyContainedIn(region));
		} else {
			TaskDataAccesses::accesses_t::iterator position =
				accessStructures._accesses.iterator_to(*dataAccess);
			position = accessStructures._accesses.fragmentByIntersection(
				position, region,
				false,
				[&](DataAccess const &toBeDuplicated) -> DataAccess * {
					assert(toBeDuplicated.isRegistered());
					return duplicateDataAccess(toBeDuplicated, accessStructures);
				},
				[&](DataAccess *fragment, DataAccess *originalDataAccess) {
					setUpNewFragment(fragment, originalDataAccess, accessStructures);
				}
			);
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			assert(dataAccess->getAccessRegion().fullyContainedIn(region));
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
	
	
	static inline void activateForcedRemovalOfBottomMapAccesses(
		Task *task, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		// For each bottom map entry
		foreachBottomMapEntry(
			accessStructures, task,
			[&] (DataAccess *access, TaskDataAccesses &currentAccessStructures, Task *currentTask) {
				assert(access->getNext() == nullptr);
				assert(access->isInBottomMap());
				assert(!access->hasBeenDiscounted());
				assert(!access->hasForcedRemoval());
				
				DataAccessStatusEffects initialStatus(access);
				access->forceRemoval();
				DataAccessStatusEffects updatedStatus(access);
				
				handleDataAccessStatusChanges(
					initialStatus, updatedStatus,
					access, currentAccessStructures, currentTask,
					hpDependencyData
				);
			},
			[&] (BottomMapEntry *bottomMapEntry) {
				// Remove the bottom map entry
				accessStructures._subaccessBottomMap.erase(bottomMapEntry);
				delete bottomMapEntry;
			}
		);
	}
	
	
	static void applyUpdateOperationOnAccess(
		UpdateOperation const &updateOperation,
		DataAccess *access, TaskDataAccesses &accessStructures,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		// Fragment if necessary
		access = fragmentAccess(access, updateOperation._region, accessStructures);
		assert(access != nullptr);
		
		DataAccessStatusEffects initialStatus(access);
		
		// Read, Write, Concurrent Satisfiability
		if (updateOperation._makeReadSatisfied) {
			access->setReadSatisfied();
		}
		if (updateOperation._makeWriteSatisfied) {
			access->setWriteSatisfied();
		}
		if (updateOperation._makeConcurrentSatisfied) {
			access->setConcurrentSatisfied();
		}
		
		// Reduction Satisfiability
		if (updateOperation._makeReductionSatisfied == any_reduction_type_and_operator) {
			access->setAnyReductionSatisfied();
		} else if (
			(updateOperation._makeReductionSatisfied != no_reduction_type_and_operator)
			&& (updateOperation._makeReductionSatisfied == access->getReductionTypeAndOperatorIndex())
		) {
			access->setMatchingReductionSatisfied();
		}
		
		// Topmost
		if (updateOperation._makeTopmost) {
			access->setTopmost();
		}
		
		DataAccessStatusEffects updatedStatus(access);
		
		handleDataAccessStatusChanges(
			initialStatus, updatedStatus,
			access, accessStructures, updateOperation._task, 
			hpDependencyData
		);
	}
	
	static void processUpdateOperation(
		UpdateOperation const &updateOperation,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		assert(!updateOperation.empty());
		TaskDataAccesses &accessStructures = updateOperation._task->getDataAccesses();
		
		if (updateOperation._toAccesses) {
			// Update over Accesses
			accessStructures._accesses.processIntersecting(
				updateOperation._region,
				[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
					DataAccess *access = &(*accessPosition);
					
					applyUpdateOperationOnAccess(updateOperation, access, accessStructures, hpDependencyData);
					
					return true;
				}
			);
		} else {
			// Update over Fragments
			accessStructures._accessFragments.processIntersecting(
				updateOperation._region,
				[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
					DataAccess *fragment = &(*fragmentPosition);
					
					applyUpdateOperationOnAccess(updateOperation, fragment, accessStructures, hpDependencyData);
					
					return true;
				}
			);
		}
	}
	
	
	static inline void processDelayedOperations(
		/* INOUT */ CPUDependencyData &hpDependencyData
	) {
		Task *lastLocked = nullptr;
		
		while (!hpDependencyData._delayedOperations.empty()) {
			UpdateOperation &delayedOperation = hpDependencyData._delayedOperations.front();
			
			assert(delayedOperation._task != nullptr);
			if (delayedOperation._task != lastLocked) {
				if (lastLocked != nullptr) {
					lastLocked->getDataAccesses()._lock.unlock();
				}
				lastLocked = delayedOperation._task;
				lastLocked->getDataAccesses()._lock.lock();
			}
			
			processUpdateOperation(delayedOperation, hpDependencyData);
			
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
		DataAccessRegion subregion,
		bool createSubregionBottomMapEntry, /* Out */ BottomMapEntry *&bottomMapEntry
	) {
		DataAccess *dataAccess = &(*accessPosition);
		assert(dataAccess != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(bottomMapEntry == nullptr);
		
		assert(!accessStructures._accessFragments.contains(dataAccess->getAccessRegion()));
		
		Instrument::data_access_id_t instrumentationId =
			Instrument::createdDataSubaccessFragment(dataAccess->getInstrumentationId());
		DataAccess *fragment = new DataAccess(
			dataAccess->getType(),
			dataAccess->isWeak(),
			dataAccess->getOriginator(),
			dataAccess->getAccessRegion(),
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
		
		// NOTE: This may in the future need to be included in the common status changes code
		dataAccess->setHasSubaccesses();
		
		if (createSubregionBottomMapEntry) {
			bottomMapEntry = new BottomMapEntry(dataAccess->getAccessRegion(), dataAccess->getOriginator(), /* Not local */ false);
			accessStructures._subaccessBottomMap.insert(*bottomMapEntry);
		} else if (subregion != dataAccess->getAccessRegion()) {
			dataAccess->getAccessRegion().processIntersectingFragments(
				subregion,
				[&](DataAccessRegion excludedSubregion) {
					bottomMapEntry = new BottomMapEntry(excludedSubregion, dataAccess->getOriginator(), /* Not local */ false);
					accessStructures._subaccessBottomMap.insert(*bottomMapEntry);
				},
				[&](__attribute__((unused)) DataAccessRegion intersection) {
					assert(!createSubregionBottomMapEntry);
				},
				[&](__attribute__((unused)) DataAccessRegion unmatchedRegion) {
					// This part is not covered by the access
				}
			);
		}
		
		return fragment;
	}
	
	
	template <typename MatchingProcessorType, typename MissingProcessorType>
	static inline bool foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRegion(
		Task *parent, TaskDataAccesses &parentAccessStructures,
		DataAccessRegion region,
		MatchingProcessorType matchingProcessor, MissingProcessorType missingProcessor,
		bool removeBottomMapEntry
	) {
		assert(parent != nullptr);
		assert((&parentAccessStructures) == (&parent->getDataAccesses()));
		assert(!parentAccessStructures.hasBeenDeleted());
		
		return parentAccessStructures._subaccessBottomMap.processIntersectingAndMissing(
			region,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				DataAccessRegion subregion = region.intersect(bottomMapEntry->getAccessRegion());
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				bool result = true;
				if (subtask != parent) {
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					result = subtaskAccessStructures._accesses.processIntersecting(
						subregion,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *previous = &(*accessPosition);
							
							assert(previous->getNext() == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(previous, subregion, subtaskAccessStructures);
							
							return matchingProcessor(previous, bottomMapEntry);
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					
					// For each fragment of the parent that matches
					result = parentAccessStructures._accessFragments.processIntersecting(
						subregion,
						[&] (TaskDataAccesses::accesses_t::iterator fragmentPosition) -> bool {
							DataAccess *previous = &(*fragmentPosition);
							
							assert(previous->getNext() == nullptr);
							assert(previous->isInBottomMap());
							assert(!previous->hasBeenDiscounted());
							
							previous = fragmentAccess(previous, subregion, parentAccessStructures);
							
							return matchingProcessor(previous, bottomMapEntry);
						}
					);
				}
				
				if (removeBottomMapEntry) {
					bottomMapEntry = fragmentBottomMapEntry(bottomMapEntry, subregion, parentAccessStructures);
					parentAccessStructures._subaccessBottomMap.erase(*bottomMapEntry);
					delete bottomMapEntry;
				}
				
				return result;
			},
			[&](DataAccessRegion missingRegion) -> bool {
				parentAccessStructures._accesses.processIntersectingAndMissing(
					missingRegion,
					[&](TaskDataAccesses::accesses_t::iterator superaccessPosition) -> bool {
						BottomMapEntry *bottomMapEntry = nullptr;
						
						DataAccessStatusEffects initialStatus;
						
						DataAccess *previous = createInitialFragment(
							superaccessPosition, parentAccessStructures,
							missingRegion, !removeBottomMapEntry, /* Out */ bottomMapEntry
						);
						assert(previous != nullptr);
						assert(previous->isFragment());
						
						previous->setTopmost();
						previous->setRegistered();
						
						DataAccessStatusEffects updatedStatus(previous);
						
						{
							CPUDependencyData hpDependencyData;
							handleDataAccessStatusChanges(
								initialStatus, updatedStatus,
								previous, parentAccessStructures, parent,
								hpDependencyData
							);
							assert(hpDependencyData.empty());
						}
						
						previous = fragmentAccess(previous, missingRegion, parentAccessStructures);
						
						return matchingProcessor(previous, bottomMapEntry);
					},
					[&](DataAccessRegion regionUncoveredByParent) -> bool {
						return missingProcessor(regionUncoveredByParent);
					}
				);
				
				return true;
			}
		);
	}
	
	
	template <typename ProcessorType, typename BottomMapEntryProcessorType>
	static inline void foreachBottomMapMatch(
		DataAccessRegion const &region,
		TaskDataAccesses &accessStructures, Task *task,
		ProcessorType processor,
		BottomMapEntryProcessorType bottomMapEntryProcessor = [] (BottomMapEntry *) {}
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		accessStructures._subaccessBottomMap.processIntersecting(
			region,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				DataAccessRegion subregion = region.intersect(bottomMapEntry->getAccessRegion());
				
				if (subtask != task) {
					// A regular access
					
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					subtaskAccessStructures._accesses.processIntersecting(
						subregion,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *subaccess = &(*accessPosition);
							assert(subaccess != nullptr);
							assert(subaccess->isReachable());
							assert(subaccess->isInBottomMap());
							assert(!subaccess->hasBeenDiscounted());
							
							subaccess = fragmentAccess(subaccess, subregion, subtaskAccessStructures);
							
							processor(subaccess, subtaskAccessStructures, subtask);
							
							return true;
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					accessStructures._accessFragments.processIntersecting(
						subregion,
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(fragment, subregion, accessStructures);
							
							processor(fragment, accessStructures, task);
							
							return true;
						}
					);
				}
				
				bottomMapEntryProcessor(bottomMapEntry);
				
				return true;
			}
		);
	}
	
	
	template <typename ProcessorType, typename BottomMapEntryProcessorType>
	static inline void foreachBottomMapEntry(
		TaskDataAccesses &accessStructures, Task *task,
		ProcessorType processor,
		BottomMapEntryProcessorType bottomMapEntryProcessor = [] (BottomMapEntry *) {}
	) {
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		accessStructures._subaccessBottomMap.processAll(
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				
				DataAccessRegion const &region = bottomMapEntry->getAccessRegion();
				
				if (subtask != task) {
					// A regular access
					
					TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
					subtaskAccessStructures._lock.lock();
					
					// For each access of the subtask that matches
					subtaskAccessStructures._accesses.processIntersecting(
						region,
						[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
							DataAccess *subaccess = &(*accessPosition);
							assert(subaccess != nullptr);
							assert(subaccess->isReachable());
							assert(subaccess->isInBottomMap());
							assert(!subaccess->hasBeenDiscounted());
							
							subaccess = fragmentAccess(subaccess, region, subtaskAccessStructures);
							
							processor(subaccess, subtaskAccessStructures, subtask);
							
							return true;
						}
					);
					
					subtaskAccessStructures._lock.unlock();
				} else {
					// A fragment
					accessStructures._accessFragments.processIntersecting(
						region,
						[&] (TaskDataAccesses::access_fragments_t::iterator fragmentPosition) -> bool {
							DataAccess *fragment = &(*fragmentPosition);
							assert(fragment != nullptr);
							assert(fragment->isReachable());
							assert(fragment->isInBottomMap());
							assert(!fragment->hasBeenDiscounted());
							
							fragment = fragmentAccess(fragment, region, accessStructures);
							
							processor(fragment, accessStructures, task);
							
							return true;
						}
					);
				}
				
				bottomMapEntryProcessor(bottomMapEntry);
				
				return true;
			}
		);
	}
	
	
	static inline void processBottomMapUpdate(
		BottomMapUpdateOperation &operation,
		TaskDataAccesses &accessStructures, Task *task,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		assert(task != nullptr);
		assert(!operation.empty());
		assert(!operation._region.empty());
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		if (operation._linkBottomMapAccessesToNext) {
			assert(!operation._activateForcedRemovalOfBottomMapAccesses);
			foreachBottomMapMatch(
				operation._region,
				accessStructures, task,
				[&] (DataAccess *access, TaskDataAccesses &currentAccessStructures, Task *currentTask) {
					DataAccessStatusEffects initialStatus(access);
					
					if (operation._inhibitReadPropagation) {
						assert(access->canPropagateReadSatisfiability());
						access->unsetCanPropagateReadSatisfiability();
					}
					
					if (operation._inhibitConcurrentPropagation) {
						assert(access->canPropagateConcurrentSatisfiability());
						access->unsetCanPropagateConcurrentSatisfiability();
					}
					
					if (operation._inhibitReductionPropagation == any_reduction_type_and_operator) {
						access->unsetCanPropagateAnyReductionSatisfiability();
					} else if (
						(operation._inhibitReductionPropagation != no_reduction_type_and_operator)
						&& (access->getReductionTypeAndOperatorIndex() == operation._inhibitReductionPropagation)
					) {
						access->unsetCanPropagateMatchingReductionSatisfiability();
					}
					
					access->setNext(operation._next);
					
					DataAccessStatusEffects updatedStatus(access);
					
					handleDataAccessStatusChanges(
						initialStatus, updatedStatus,
						access, currentAccessStructures, currentTask,
						hpDependencyData
					);
				},
				[] (BottomMapEntry *) {}
			);
		} else if (operation._activateForcedRemovalOfBottomMapAccesses) {
			foreachBottomMapMatch(
				operation._region,
				accessStructures, task,
				[&] (DataAccess *access, TaskDataAccesses &currentAccessStructures, Task *currentTask) {
					DataAccessStatusEffects initialStatus(access);
					access->forceRemoval();
					DataAccessStatusEffects updatedStatus(access);
					
					handleDataAccessStatusChanges(
						initialStatus, updatedStatus,
						access, currentAccessStructures, currentTask,
						hpDependencyData
					);
				},
				[] (BottomMapEntry *) {}
			);
		}
	}
	
	
	static inline void replaceMatchingInBottomMapLinkAndPropagate(
		Task *task,  TaskDataAccesses &accessStructures,
		DataAccessRegion region,
		Task *parent, TaskDataAccesses &parentAccessStructures,
		/* inout */ CPUDependencyData &hpDependencyData
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!parentAccessStructures.hasBeenDeleted());
		
		bool local = false;
		#ifndef NDEBUG
			bool lastWasLocal = false;
			bool first = true;
		#endif
		
		// Link accesses to their corresponding predecessor
		foreachBottomMapMatchPossiblyCreatingInitialFragmentsAndMissingRegion(
			parent, parentAccessStructures,
			region,
			[&](DataAccess *previous, BottomMapEntry *bottomMapEntry) -> bool {
				assert(previous != nullptr);
				assert(previous->isReachable());
				assert(!previous->hasBeenDiscounted());
				assert(previous->getNext() == nullptr);
				assert(!previous->hasForcedRemoval());
				
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
				assert(previous->getAccessRegion().fullyContainedIn(region));
				
				DataAccessStatusEffects initialStatus(previous);
				
				// Link the dataAccess
				previous->setNext(task);
				previous->unsetInBottomMap();
				
				DataAccessStatusEffects updatedStatus(previous);
				
				handleDataAccessStatusChanges(
					initialStatus, updatedStatus,
					previous, previousAccessStructures, previousTask,
					hpDependencyData
				);
				
				return true;
			},
			[&](DataAccessRegion missingRegion) -> bool {
				assert(!parentAccessStructures._accesses.contains(missingRegion));
				
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
					missingRegion,
					[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
						DataAccess *targetAccess = &(*position);
						assert(targetAccess != nullptr);
						assert(!targetAccess->hasBeenDiscounted());
						
						targetAccess = fragmentAccess(targetAccess, missingRegion, accessStructures);
						
						DataAccessStatusEffects initialStatus(targetAccess);
						targetAccess->setReadSatisfied();
						targetAccess->setWriteSatisfied();
						targetAccess->setConcurrentSatisfied();
						targetAccess->setAnyReductionSatisfied();
						targetAccess->setMatchingReductionSatisfied();
						targetAccess->setTopmost();
						DataAccessStatusEffects updatedStatus(targetAccess);
						
						handleDataAccessStatusChanges(
							initialStatus, updatedStatus,
							targetAccess, accessStructures, task,
							hpDependencyData
						);
						
						return true;
					}
				);
				
				return true;
			},
			true /* Erase the entry from the bottom map */
		);
		
		// Add the entry to the bottom map
		BottomMapEntry *bottomMapEntry = new BottomMapEntry(region, task, local);
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
				
				DataAccessStatusEffects initialStatus(dataAccess);
				dataAccess->setNewInstrumentationId(task->getInstrumentationTaskId());
				dataAccess->setInBottomMap();
				dataAccess->setRegistered();
#ifndef NDEBUG
				dataAccess->setReachable();
#endif
				DataAccessStatusEffects updatedStatus(dataAccess);
				
				handleDataAccessStatusChanges(
					initialStatus, updatedStatus,
					dataAccess, accessStructures, task,
					hpDependencyData
				);
				
				// Unlock to avoid potential deadlock
				accessStructures._lock.unlock();
				
				replaceMatchingInBottomMapLinkAndPropagate(
					task, accessStructures,
					dataAccess->getAccessRegion(),
					parent, parentAccessStructures,
					hpDependencyData
				);
				
				// Relock to advance the iterator
				accessStructures._lock.lock();
				
				return true;
			}
		);
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
				
				DataAccessStatusEffects initialStatus(fragment);
				fragment->setComplete();
				DataAccessStatusEffects updatedStatus(fragment);
				
				handleDataAccessStatusChanges(
					initialStatus, updatedStatus,
					fragment, accessStructures, task,
					hpDependencyData
				);
				
				return true;
			}
		);
	}
	
	
	template <typename ProcessorType>
	static inline void applyToAccessAndFragments(
		DataAccess *dataAccess, DataAccessRegion const &region,
		TaskDataAccesses &accessStructures,
		ProcessorType processor
	) {
		// Fragment if necessary
		dataAccess = fragmentAccess(dataAccess, region, accessStructures);
		assert(dataAccess != nullptr);
		
		bool hasSubaccesses = dataAccess->hasSubaccesses();
		DataAccessRegion finalRegion = dataAccess->getAccessRegion();
		bool alsoSubaccesses = processor(dataAccess);
		
		if (alsoSubaccesses && hasSubaccesses) {
			accessStructures._accessFragments.processIntersecting(
				finalRegion,
				[&](TaskDataAccesses::access_fragments_t::iterator position) -> bool {
					DataAccess *fragment = &(*position);
					assert(fragment != nullptr);
					assert(fragment->isFragment());
					assert(!fragment->hasBeenDiscounted());
					
					fragment = fragmentAccess(fragment, finalRegion, accessStructures);
					assert(fragment != nullptr);
					
					processor(fragment);
					
					return true;
				}
			);
		}
	}
	
	
	static inline void finalizeAccess(
		Task *finishedTask, DataAccess *dataAccess, DataAccessRegion region,
		/* OUT */ CPUDependencyData &hpDependencyData
	) {
		assert(finishedTask != nullptr);
		assert(dataAccess != nullptr);
		
		assert(dataAccess->getOriginator() == finishedTask);
		assert(!region.empty());
		
		// The access may already have been released through the "release" directive
		if (dataAccess->complete()) {
			return;
		}
		assert(!dataAccess->hasBeenDiscounted());
		
		applyToAccessAndFragments(
			dataAccess, region,
			finishedTask->getDataAccesses(),
			[&] (DataAccess *accessOrFragment) -> bool {
				assert(!accessOrFragment->complete());
				
				DataAccessStatusEffects initialStatus(accessOrFragment);
				accessOrFragment->setComplete();
				DataAccessStatusEffects updatedStatus(accessOrFragment);
				
				handleDataAccessStatusChanges(
					initialStatus, updatedStatus,
					accessOrFragment, finishedTask->getDataAccesses(), finishedTask,
					hpDependencyData
				);
				
				return true; // Apply also to subaccesses if any
			}
		);
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
	//! \param[in] region the region of data covered by the access
	//! \param[in] reductionTypeAndOperatorIndex an index that identifies the type and the operation of the reduction
	static inline void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRegion region, reduction_type_and_operator_index_t reductionTypeAndOperatorIndex
	) {
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		accessStructures._accesses.fragmentIntersecting(
			region,
			[&](DataAccess const &toBeDuplicated) -> DataAccess * {
				assert(!toBeDuplicated.isRegistered());
				return duplicateDataAccess(toBeDuplicated, accessStructures);
			},
			[](DataAccess *, DataAccess *) {}
		);
		
		accessStructures._accesses.processIntersectingAndMissing(
			region,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *oldAccess = &(*position);
				assert(oldAccess != nullptr);
				
				upgradeAccess(oldAccess, accessType, weak, reductionTypeAndOperatorIndex);
				
				return true;
			},
			[&](DataAccessRegion missingRegion) -> bool {
				DataAccess *newAccess = createAccess(task, accessType, weak, missingRegion, false, reductionTypeAndOperatorIndex);
				
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
					bottomMapEntry->getAccessRegion(),
					[&](TaskDataAccesses::accesses_t::iterator position2) -> bool {
						DataAccess *dataAccess = &(*position2);
						assert(dataAccess != nullptr);
						assert(dataAccess->getNext() == nullptr);
						
						if (dataAccess->hasForcedRemoval()) {
							return true;
						}
						
						dataAccess = fragmentAccess(dataAccess, bottomMapEntry->getAccessRegion(), subtaskAccessStructures);
						
						DataAccessStatusEffects initialStatus(dataAccess);
						dataAccess->forceRemoval();
						DataAccessStatusEffects updatedStatus(dataAccess);
						
						handleDataAccessStatusChanges(
							initialStatus, updatedStatus,
							dataAccess, subtaskAccessStructures, subtask,
							hpDependencyData
						);
						
						return true;
					}
				);
				
				return true;
			}
		);
	}
	
	
	
	static inline void releaseAccessRegion(
		Task *task, DataAccessRegion region,
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
				region,
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					assert(dataAccess->getType() == accessType);
					assert(dataAccess->isWeak() == weak);
					
					finalizeAccess(task, dataAccess, region, /* OUT */ hpDependencyData);
					
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
					
					finalizeAccess(task, dataAccess, dataAccess->getAccessRegion(), /* OUT */ hpDependencyData);
					
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
					
#ifndef NDEBUG
					DataAccessStatusEffects currentStatus(dataAccess);
					assert(currentStatus._isRemovable);
#endif
					
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

#pragma GCC visibility pop


#endif // DATA_ACCESS_REGISTRATION_HPP
