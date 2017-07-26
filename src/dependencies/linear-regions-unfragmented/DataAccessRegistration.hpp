/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP


#include <cassert>
#include <deque>
#include <mutex>

#include "CPUDependencyData.hpp"
#include "DataAccess.hpp"
#include "DataAccessImplementation.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentTaskId.hpp>


class DataAccessRegistration {
private:
	static inline void replaceOutdatedMapProjection(
		LinearRegionDataAccessMap *map, DataAccessRange const &range,
		DataAccess *outdatedContent, DataAccess *replacement
	) {
		assert(outdatedContent != nullptr);
		assert(outdatedContent->_lock != nullptr);
		assert(outdatedContent->_lock->isLockedByThisThread());
		
		map->processIntersecting(
			range,
			[&](LinearRegionDataAccessMap::iterator position) -> bool {
				if (position->_access == outdatedContent) {
					auto intersectingFragmentPosition = map->fragmentByIntersection(position, range, false);
					
					assert(intersectingFragmentPosition != map->end());
					intersectingFragmentPosition->_access = replacement;
				}
				
				return true;
			}
		);
	}
	
	
	static inline void replaceOutdatedMapProjection(
		LinearRegionDataAccessMap *map, DataAccess *outdatedContent,
		LinearRegionDataAccessMap &replacement
	) {
		assert(outdatedContent != nullptr);
		assert(outdatedContent->_lock != nullptr);
		assert(outdatedContent->_lock->isLockedByThisThread());
		
		// NOTE: This could be implemented as a "merge"-like operation
		for (auto replacementPosition = replacement.begin(); replacementPosition != replacement.end(); replacementPosition++) {
			replaceOutdatedMapProjection(
				map, replacementPosition->_accessRange,
				outdatedContent, replacementPosition->_access
			);
		}
	}
	
	
	static void propagateSuperAccessAndBottomMap(DataAccess *dataAccess, Instrument::task_id_t triggererTaskInstrumentationId)
	{
		assert(dataAccess != nullptr);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		
		DataAccess *superAccess = dataAccess->_superAccess;
		LinearRegionDataAccessMap *bottomMap = dataAccess->_bottomMap;
		assert(bottomMap != nullptr);
		
		for (auto &nextLink : dataAccess->_next) {
			DataAccess *next = nextLink._access;
			assert(next != nullptr);
			
			// Since the accesses form a graph, we may already have visited a given one
			if (next->_superAccess == superAccess) {
				assert(next->_bottomMap == bottomMap);
				continue;
			}
			
			Instrument::reparentedDataAccess(
				next->_superAccess->_instrumentationId,
				(superAccess != nullptr ? superAccess->_instrumentationId : Instrument::data_access_id_t()),
				next->_instrumentationId,
				triggererTaskInstrumentationId
			);
			
			next->_superAccess = superAccess;
			next->_bottomMap = bottomMap;
			
			propagateSuperAccessAndBottomMap(next, triggererTaskInstrumentationId);
		}
	}
	
	
	static inline void replaceSuperAccessAndBottomMapOfSubaccesses(DataAccess *dataAccess, LinearRegionDataAccessMap *newBottomMap, Instrument::task_id_t triggererTaskInstrumentationId)
	{
		assert(dataAccess != nullptr);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		
		DataAccess *newSuperAccess = dataAccess->_superAccess;
		for (auto const &subaccessNode : dataAccess->_topSubaccesses) {
			DataAccess *subaccess = subaccessNode._access;
			assert(subaccess != nullptr);
			
			// Since the subaccesses form a graph, we may already have visited a given subaccess
			if (subaccess->_superAccess == newSuperAccess) {
				assert(subaccess->_bottomMap == newBottomMap);
				continue;
			}
			
			assert(subaccess->_superAccess == dataAccess);
			assert(subaccess->_bottomMap == &dataAccess->_bottomSubaccesses);
			
			Instrument::reparentedDataAccess(
				dataAccess->_instrumentationId,
				(newSuperAccess != nullptr ? newSuperAccess->_instrumentationId : Instrument::data_access_id_t()),
				subaccess->_instrumentationId,
				triggererTaskInstrumentationId
			);
			
			subaccess->_superAccess = newSuperAccess;
			subaccess->_bottomMap = newBottomMap;
			
			propagateSuperAccessAndBottomMap(subaccess, triggererTaskInstrumentationId);
		}
			
	}
	
	
	static inline void removeMapProjection(LinearRegionDataAccessMap *map, DataAccess *dataAccess)
	{
		assert(dataAccess != nullptr);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		
		map->processIntersecting(
			dataAccess->_range,
			[&](LinearRegionDataAccessMap::iterator position) -> bool {
				if (position->_access == dataAccess) {
					map->erase(position);
				}
				
				return true;
			}
		);
	}
	
	
	static inline void linkExistingDataAccesses(
		DataAccess *source, DataAccessRange const &sourceRange,
		DataAccess *target, DataAccessRange const &targetRange,
		bool countUnsatisfiabilityAsBlocker,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(source != nullptr);
		assert(target != nullptr);
		assert(!sourceRange.empty());
		assert(!targetRange.empty());
		assert(source->_lock != nullptr);
		assert(source->_lock->isLockedByThisThread());
		assert(source->_lock == target->_lock);
		
		DataAccessRange intersection = targetRange.intersect(sourceRange);
		assert(!intersection.empty());
		
		bool newLinkSatisfied = DataAccess::evaluateSatisfiability(source, target->_type);
		if (countUnsatisfiabilityAsBlocker && !newLinkSatisfied) {
			target->_blockerCount++;
		}
		
		source->fullLinkTo(intersection, target, !newLinkSatisfied, triggererTaskInstrumentationId);
	}
	
	
	static inline void handleLinksToNext(
		DataAccess *dataAccess, DataAccess *next,
		DataAccessRange const &nextRange,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		assert(next != nullptr);
		assert(dataAccess->_lock == next->_lock);
		
		dataAccess->_bottomSubaccesses.processIntersectingAndMissing(
			nextRange,
			// Bottom subaccesses that have an intersection with "next"
			[&](LinearRegionDataAccessMap::iterator bottomIntersectingPosition) -> bool {
				linkExistingDataAccesses(
					bottomIntersectingPosition->_access, bottomIntersectingPosition->_accessRange,
					next, nextRange,
					true /* Count unsatisfied links as blockers */,
					triggererTaskInstrumentationId
				);
				
				return true;
			},
			// Ranges of "next" not covered by any bottom subaccess
			[&](DataAccessRange const &unmatchedNextFragment) -> bool {
				dataAccess->_previous.processIntersecting(
					unmatchedNextFragment,
					// The previous of the DataAccess to be removed constrained to the "holes" and the same graph
					[&](DataAccessPreviousLinks::iterator previousPosition) -> bool {
						DataAccess *previous = previousPosition->_access;
						assert(previous != nullptr);
						
						DataAccessRange const &previousRange = previousPosition->_accessRange;
						assert(!unmatchedNextFragment.intersect(previousRange).empty());
						
						linkExistingDataAccesses(
							previous, previousRange,
							next, nextRange,
							true /* Do consider satisfiability */,
							triggererTaskInstrumentationId
						);
						
						return true;
					}
				);
				
				return true;
			}
		);
	}
	
	
	static inline void handleLinksToPrevious(
		DataAccess *dataAccess, DataAccess *previous,
		DataAccessRange const &previousRange,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		assert(previous != nullptr);
		assert(dataAccess->_lock == previous->_lock);
		
		dataAccess->_topSubaccesses.processIntersecting(
			previousRange,
			// Top subaccesses that have an intersection with "previous"
			[&](LinearRegionDataAccessMap::iterator intersectingPosition) -> bool {
				linkExistingDataAccesses(
					previous, previousRange,
					intersectingPosition->_access, intersectingPosition->_accessRange,
					false /* Do not count unsatisfied links as blockers, since they were already counted in the effective previous relation */,
					triggererTaskInstrumentationId
				);
				
				return true;
			}
		);
	}
	
	
	static void propagateSatisfiabilityChangeToNext(
		DataAccessRange const &range, DataAccess *fromDataAccess,
		DataAccessNextLinks::iterator &nextLink,
		CPUDependencyData::satisfied_originator_list_t /* OUT */ &satisfiedOriginators,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(fromDataAccess != nullptr);
		assert(fromDataAccess->_blockerCount == 0);
		assert(fromDataAccess->_lock != nullptr);
		assert(fromDataAccess->_lock->isLockedByThisThread());
		
		DataAccess *nextAccess = nextLink->_access;
		assert(nextAccess != nullptr);
		assert(!nextLink->_satisfied);
		assert(nextAccess->_blockerCount != 0);
		assert(fromDataAccess->_lock == nextAccess->_lock);
		
		bool linkBecomesSatisfied = DataAccess::evaluateSatisfiability(fromDataAccess, nextAccess->_type);
		if (linkBecomesSatisfied) {
			nextLink->_satisfied = true;
			
			nextAccess->_blockerCount--;
			assert(nextAccess->_blockerCount >= 0);
			
			if (nextAccess->_blockerCount == 0) {
				Instrument::dataAccessBecomesSatisfied(
					nextAccess->_instrumentationId,
					false, false, true,
					triggererTaskInstrumentationId,
					nextAccess->_originator->getInstrumentationTaskId()
				);
				
				if (!nextAccess->_weak) {
					satisfiedOriginators.push_back(nextAccess->_originator);
				}
				
				DataAccessRange effectiveRange = range.intersect(nextAccess->_range);
				propagateSatisfiabilityChange(effectiveRange, nextAccess, satisfiedOriginators, triggererTaskInstrumentationId);
			}
		}
	}
	
	
	static void propagateSatisfiabilityChangeToSubaccess(
		DataAccessRange const &range, DataAccess *fromDataAccess,
		LinearRegionDataAccessMap::iterator &subaccessPosition,
		CPUDependencyData::satisfied_originator_list_t /* OUT */ &satisfiedOriginators,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(fromDataAccess != nullptr);
		assert(fromDataAccess->_blockerCount == 0);
		assert(fromDataAccess->_lock != nullptr);
		assert(fromDataAccess->_lock->isLockedByThisThread());
		
		DataAccess *subaccess = subaccessPosition->_access;
		assert(subaccess != nullptr);
		assert(fromDataAccess->_lock == subaccess->_lock);
		
		bool linkBecomesSatisfied = fromDataAccess->propagatesSatisfiability(subaccess->_type);
		if (linkBecomesSatisfied) {
			subaccess->_blockerCount--;
			assert(subaccess->_blockerCount >= 0);
			
			DataAccessRange effectiveRange = range.intersect(subaccess->_range);
			
			if (subaccess->_blockerCount == 0) {
				Instrument::dataAccessBecomesSatisfied(
					subaccess->_instrumentationId,
					false, false, true,
					triggererTaskInstrumentationId,
					subaccess->_originator->getInstrumentationTaskId()
				);
				
				if (!subaccess->_weak) {
					satisfiedOriginators.push_back(subaccess->_originator);
				}
				propagateSatisfiabilityChange(effectiveRange, subaccess, satisfiedOriginators, triggererTaskInstrumentationId);
			}
		}
	}
	
	
	static void propagateSatisfiabilityChange(
		DataAccessRange const &range, DataAccess *fromDataAccess,
		CPUDependencyData::satisfied_originator_list_t /* OUT */ &satisfiedOriginators,
		Instrument::task_id_t triggererTaskInstrumentationId
	) {
		assert(fromDataAccess != nullptr);
		assert(fromDataAccess->_blockerCount == 0);
		assert(fromDataAccess->_lock != nullptr);
		assert(fromDataAccess->_lock->isLockedByThisThread());
		
		fromDataAccess->_next.processIntersecting(
			range,
			[&](DataAccessNextLinks::iterator nextLink) -> bool {
				if (!nextLink->_satisfied) {
					propagateSatisfiabilityChangeToNext(range, fromDataAccess, nextLink, /* OUT */ satisfiedOriginators, triggererTaskInstrumentationId);
				}
				
				return true;
			}
		);
		
		fromDataAccess->_topSubaccesses.processIntersecting(
			range,
			[&](LinearRegionDataAccessMap::iterator subaccessPosition) -> bool {
				propagateSatisfiabilityChangeToSubaccess(range, fromDataAccess, subaccessPosition, /* OUT */ satisfiedOriginators, triggererTaskInstrumentationId);
				
				return true;
			}
		);
	}
	
	
	static inline void unregisterDataAccess(
		Instrument::task_id_t instrumentationTaskId,
		DataAccess *dataAccess,
		LinearRegionDataAccessMap *topMap, LinearRegionDataAccessMap *bottomMap,
		CPUDependencyData::satisfied_originator_list_t /* OUT */ &satisfiedOriginators
	) {
		assert(dataAccess != nullptr);
		assert((dataAccess->_blockerCount == 0) || dataAccess->_weak);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		
		// Update the top map with any projection of the top subaccesses that will become unobstructed
		if (topMap != nullptr) {
			replaceOutdatedMapProjection(topMap, dataAccess, dataAccess->_topSubaccesses);
		}
		
		// Update the bottom map with any projection of the bottom subaccesses that will become unobstructed
		assert(bottomMap != nullptr);
		replaceOutdatedMapProjection(bottomMap, dataAccess, dataAccess->_bottomSubaccesses);
		
		replaceSuperAccessAndBottomMapOfSubaccesses(dataAccess, bottomMap, instrumentationTaskId);
		
		// Remove links from "previous" objects (half link) since they will be replaced in the next loop
		for (auto previousLinkPosition = dataAccess->_previous.begin(); previousLinkPosition != dataAccess->_previous.end(); previousLinkPosition++) {
			DataAccessRange const &previousRange = previousLinkPosition->_accessRange;
			DataAccess *previous = previousLinkPosition->_access;
			
			// Check that there is a correct reverse link
			assert(previous != nullptr);
			assert(dataAccess->_lock == previous->_lock);
			assert(previous->_next.find(previousRange) != previous->_next.end());
			assert(previous->_next.find(previousRange)->_access == dataAccess);
			
			// Erase the link from the DataAccess to be removed
			{
				auto reverseLinkPosition = previous->_next.find(previousRange);
				assert(reverseLinkPosition != previous->_next.end());
				assert(reverseLinkPosition->_accessRange == previousRange);
				assert(reverseLinkPosition->_access == dataAccess);
				assert(reverseLinkPosition->_satisfied || dataAccess->_weak);
				assert(dataAccess->_originator != nullptr);
				
				Instrument::unlinkedDataAccesses(
					previous->_instrumentationId,
					dataAccess->_originator->getInstrumentationTaskId(),
					true /* direct link */,
					instrumentationTaskId
				);
				
				previous->_next.erase(reverseLinkPosition);
			}
		}
		
		// Remove (back) links (from) next and (fully) link them back to their new previous
		for (auto nextLinkPosition = dataAccess->_next.begin(); nextLinkPosition != dataAccess->_next.end(); nextLinkPosition++) {
			DataAccessRange const &nextRange = nextLinkPosition->_accessRange;
			DataAccess *next = nextLinkPosition->_access;
			
			// Check that there is a correct reverse link
			assert(next != nullptr);
			assert(dataAccess->_lock == next->_lock);
			assert(next->_previous.find(nextRange) != next->_previous.end());
			assert(!next->_previous.find(nextRange)->getAccessRange().intersect(nextRange).empty());
			assert(next->_previous.find(nextRange)->_access == dataAccess);
			
			int originalBlockerCount = next->_blockerCount;
			
			// Erase the link from the DataAccess to be removed
			{
				auto reverseLinkPosition = next->_previous.find(nextRange);
				assert(reverseLinkPosition != next->_previous.end());
				assert(reverseLinkPosition->_accessRange == nextRange);
				assert(reverseLinkPosition->_access == dataAccess);
				assert(next->_originator != nullptr);
				
				Instrument::unlinkedDataAccesses(
					dataAccess->_instrumentationId,
					next->_originator->getInstrumentationTaskId(),
					true /* direct link */,
					instrumentationTaskId
				);
				
				next->_previous.erase(reverseLinkPosition);
			}
			
			if (!nextLinkPosition->_satisfied) {
				next->_blockerCount--;
			}
			assert(next->_blockerCount >= 0);
			
			// Update the top map with any projection of "next" that becomes unobstructed
			if (topMap != nullptr) {
				replaceOutdatedMapProjection(topMap, nextRange, dataAccess, next);
			}
			
			// Link bottom subaccesses and previous as necessary to "next". Also increase the blocker count
			// due to effective previous at outer levels
			handleLinksToNext(dataAccess, next, nextRange, instrumentationTaskId);
			
			// Check if the access becomes satisfied
			assert(next->_blockerCount >= 0);
			if ((originalBlockerCount > 0) && (next->_blockerCount == 0)) {
				Task *nextOriginator = next->_originator;
				assert(nextOriginator != nullptr);
				
				Instrument::dataAccessBecomesSatisfied(
					next->_instrumentationId,
					false, false, true,
					instrumentationTaskId,
					next->_originator->getInstrumentationTaskId()
				);
				
				if (!next->_weak) {
					satisfiedOriginators.push_back(nextOriginator);
				}
				propagateSatisfiabilityChange(nextRange, next, satisfiedOriginators, instrumentationTaskId);
			}
		}
		
		// Process effects over the "previous" objects
		for (auto previousLinkPosition = dataAccess->_previous.begin(); previousLinkPosition != dataAccess->_previous.end(); previousLinkPosition++) {
			DataAccessRange const &previousRange = previousLinkPosition->_accessRange;
			DataAccess *previous = previousLinkPosition->_access;
			assert(dataAccess->_lock == previous->_lock);
			
			// Update the bottom map with any projection of "previous" that becomes unobstructed
			replaceOutdatedMapProjection(bottomMap, previousRange, dataAccess, previous);
			
			// Link top subaccesses as necessary to "previous"
			handleLinksToPrevious(dataAccess, previous, previousRange, instrumentationTaskId);
			
			// The removal of the DataAccess does not affect the satisfiability of the previous accesses
		}
		
		// Remove in the top and bottom maps any lingering range still covered by the DataAccess to be removed
		if (topMap != nullptr) {
			removeMapProjection(topMap, dataAccess);
		}
		removeMapProjection(bottomMap, dataAccess);
		
		Instrument::removedDataAccess(
			dataAccess->_instrumentationId,
			instrumentationTaskId
		);
		
		// Remove the other half of the links and back links that is sourced from dataAccess
		dataAccess->_next.clear();
		dataAccess->_previous.clear();
	}
	
	
	//! Process all the originators for whose a DataAccess has become satisfied
	static inline void processSatisfiedOriginators(CPUDependencyData::satisfied_originator_list_t &satisfiedOriginators, ComputePlace *computePlace)
	{
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			bool becomesReady = satisfiedOriginator->decreasePredecessors();
			if (becomesReady) {
				ComputePlace *idleComputePlace = Scheduler::addReadyTask(satisfiedOriginator, computePlace, SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT);
				
				if (idleComputePlace != nullptr) {
					ThreadManager::resumeIdle((CPU *) idleComputePlace);
				}
			}
		}
	}
	
	
	static inline void registerTaskDataAccessFragment(
		Task *task, DataAccessType accessType, bool weak,
		DataAccess *superAccess, SpinLock *lock, LinearRegionDataAccessMap *bottomMap,
		LinearRegionDataAccessMap::iterator positionOfPreviousAccess
	) {
		assert(positionOfPreviousAccess != bottomMap->end());
		
		DataAccessRange intersectingFragment = positionOfPreviousAccess->_accessRange;
		DataAccess *previousAccess = positionOfPreviousAccess->_access;
		assert(previousAccess != nullptr);
		assert(previousAccess->_lock != nullptr);
		assert(previousAccess->_lock == lock);
		assert(previousAccess->_lock->isLockedByThisThread());
		
		// A new data access, as opposed to a repeated or upgraded one
		assert(previousAccess->_originator != task);
		
		// Find its satisfiability
		bool satisfied = DataAccess::evaluateSatisfiability(previousAccess, accessType);
		
		// Create a corresponding fragment
		Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
			(superAccess != nullptr ? superAccess->_instrumentationId : Instrument::data_access_id_t()),
			accessType, weak, intersectingFragment,
			false, false, (satisfied ? 0 : 1),
			task->getInstrumentationTaskId()
		);
		DataAccess *dataAccess = new DataAccess(
			superAccess, lock, bottomMap,
			accessType, weak, (satisfied ? 0 : 1),
			task, intersectingFragment,
			dataAccessInstrumentationId
		);
		
		previousAccess->fullLinkTo(intersectingFragment, dataAccess, !satisfied, task->getInstrumentationTaskId());
		task->getDataAccesses().push_back(*dataAccess);
		if (!satisfied && ! weak) {
			task->increasePredecessors();
		}
		
		// Replace the old DataAccess by the new one
		positionOfPreviousAccess->_access = dataAccess;
	}
	
	
	static inline void handleFullDataAccessUpgrade(
		Task *task, DataAccessType accessType, bool weak, LinearRegionDataAccessMap::iterator accessPosition
	) {
		assert(task != nullptr);
		
		// Access Upgrade Phase 2: Special handling for fully overlapping upgrades
		DataAccess *dataAccess = accessPosition->_access;
		assert(dataAccess != nullptr);
		assert(dataAccess->_originator == task);
		assert(dataAccess->_lock != nullptr);
		assert(dataAccess->_lock->isLockedByThisThread());
		
		bool doesNotBecomeUnsatisfied = DataAccess::upgradeAccess(task, dataAccess, accessType, weak);
		
		if (!doesNotBecomeUnsatisfied && !dataAccess->_weak) {
			task->increasePredecessors();
		}
	}
	
	
	static inline void handleFullDataAccessUpgrade(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange accessRange, LinearRegionDataAccessMap *bottomMap
	) {
		assert(task != nullptr);
		assert(bottomMap != nullptr);
		
		LinearRegionDataAccessMap::iterator accessPosition = bottomMap->find(accessRange);
		assert(accessPosition != bottomMap->end());
		assert(accessPosition->_accessRange == accessRange);
		
		handleFullDataAccessUpgrade(task, accessType, weak, accessPosition);
	}
	
	
	static inline void handleFullAndPartialDataAccessUpgrade(
		LinearRegionDataAccessMap::iterator oldAccessPosition,
		Task *task, DataAccessType accessType, bool weak, DataAccessRange accessRange,
		DataAccess *superAccess, SpinLock *lock,
		LinearRegionDataAccessMap *topMap, LinearRegionDataAccessMap *bottomMap
	) {
		DataAccess *oldDataAccess = oldAccessPosition->_access;
		assert(oldDataAccess != nullptr);
		
		DataAccessRange oldAccessRange = oldAccessPosition->_accessRange;
		DataAccessType oldAccessType = oldDataAccess->_type;
		bool oldWeak = oldDataAccess->_weak;
		assert(oldDataAccess->_lock == lock);
		assert(oldDataAccess->_superAccess == superAccess);
		assert(oldDataAccess->_bottomMap == bottomMap);
		assert(lock->isLockedByThisThread());
		
		// Simple case: perfectly matching upgrade
		if (oldAccessRange == accessRange) {
			handleFullDataAccessUpgrade(task, accessType, weak, oldAccessPosition);
			return;
		}
		
		// Remove the old access from the task itself
		{
			TaskDataAccesses &taskAccesses = task->getDataAccesses();
			auto oldAccessInTaskPosition = taskAccesses.iterator_to(*oldDataAccess);
			assert(oldAccessInTaskPosition != taskAccesses.end());
			taskAccesses.erase(oldAccessInTaskPosition);
		}
		
		// Unlink the old access, remove it from the map and delete it
		{
			CPUDependencyData::satisfied_originator_list_t dummyList;
			unregisterDataAccess(
				task->getInstrumentationTaskId(),
				oldDataAccess, topMap, bottomMap,
				dummyList
			);
			assert(dummyList.empty());
			delete oldDataAccess;
		}
		
		// Add back each fragment of the old access and add each new fragment (which may trigger perfectly overlapping updates)
		accessRange.processIntersectingFragments(
			oldAccessRange,
			// A fragment in the current DataAccessRange that does not need to be upgraded
			[&](DataAccessRange const &fragment) {
				registerTaskDataAccessPrelocked(task, accessType, weak, fragment, superAccess, lock, topMap, bottomMap, false);
			},
			// A fragment that must be upgraded
			[&](DataAccessRange const &fragment) {
				registerTaskDataAccessPrelocked(task, oldAccessType, oldWeak, fragment, superAccess, lock, topMap, bottomMap, false);
				handleFullDataAccessUpgrade(task, accessType, weak, fragment, bottomMap);
			},
			// A fragment from a previous access that is not covered in the curent DataAccessRange
			[&](DataAccessRange const &fragment) {
				registerTaskDataAccessPrelocked(task, oldAccessType, oldWeak, fragment, superAccess, lock, topMap, bottomMap, false);
			}
		);
	}
	
	
	static inline void registerTaskDataAccessPrelocked(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange accessRange,
		DataAccess *superAccess, SpinLock *lock,
		LinearRegionDataAccessMap *topMap, LinearRegionDataAccessMap *bottomMap,
		bool checkUpgrades = true
	) {
		assert(task != nullptr);
		assert(lock != nullptr);
		assert(bottomMap != nullptr);
		assert(lock->isLockedByThisThread());
		assert((bottomMap->_superAccess == nullptr) || (bottomMap->_superAccess->_lock == lock));
		
		//
		// Handling of access upgrades
		//
		
		// Access Upgrade Phase 1: Fragment the previously existing DataAccesses as necessary to avoid partially overlapping upgrades
		if (checkUpgrades) {
			bool containsUpgrades = false;
			bottomMap->processIntersecting(
				accessRange,
				[&](LinearRegionDataAccessMap::iterator position) -> bool {
					DataAccess *intersectingDataAccess = position->_access;
					assert(intersectingDataAccess != nullptr);
					
					if (intersectingDataAccess->_originator == task) {
						containsUpgrades = true;
						
						handleFullAndPartialDataAccessUpgrade(position, task, accessType, weak, accessRange, superAccess, lock, topMap, bottomMap);
					}
					
					return true;
				}
			);
			if (containsUpgrades) {
				// The calls to handleFullAndPartialDataAccessUpgrade have already added the access in a fragmented way
				return;
			}
		}
		
		
		//
		// Regular Cases
		//
		
		// Instrumentation
		Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
			(superAccess != nullptr ? superAccess->_instrumentationId : Instrument::data_access_id_t()),
			accessType, weak,
			accessRange,
			false, false, false, // Not satisfied
			task->getInstrumentationTaskId()
		);
		
		// Create the DataAccess
		DataAccess *dataAccess = new DataAccess(
			superAccess, lock, bottomMap,
			accessType, weak, 0,
			task, accessRange,
			dataAccessInstrumentationId
		);
		
		
		bottomMap->processIntersectingAndMissing(
			accessRange,
			// Handle regions in the map that intersect
			[&](LinearRegionDataAccessMap::iterator intersectingPosition) -> bool {
				assert(intersectingPosition != bottomMap->end());
				
				DataAccess *intersectingDataAccess = intersectingPosition->_access;
				assert(intersectingDataAccess != nullptr);
				DataAccessRange const &insersectingFragment = intersectingPosition->_accessRange.intersect(accessRange);
				
				// Link and count blockers
				bool satisfied = DataAccess::evaluateSatisfiability(intersectingDataAccess, accessType);
				intersectingDataAccess->fullLinkTo(insersectingFragment, dataAccess, !satisfied, task->getInstrumentationTaskId());
				if (!satisfied) {
					dataAccess->_blockerCount++;
				}
				
				// Fragment partial overlaps and remove the fully contained fragment
				bottomMap->fragmentByIntersection(intersectingPosition, accessRange, /* remove instersection */ true);
				
				return true;
			},
			// Handle fragments not in the map (holes)
			[&](DataAccessRange const &missingFragment) -> bool {
				// Increase the blocker count due to effective previous accesses that are in an outer scope
				if (superAccess != nullptr) {
					superAccess->processEffectivePrevious(
						missingFragment,
						true, // Process direct previous too
						[&](DataAccessPreviousLinks::iterator effectivePreviousPosition) -> bool {
							DataAccess *effectivePrevious = effectivePreviousPosition->_access;
							assert(effectivePrevious != nullptr);
							
							bool satisfied = DataAccess::evaluateSatisfiability(effectivePrevious, accessType);
							if (!satisfied) {
								dataAccess->_blockerCount++;
							}
							
							return true;
						}
					);
				}
				
				if (topMap != nullptr) {
					topMap->insert(LinearRegionDataAccessMapNode(missingFragment, dataAccess));
				}
				
				return true;
			}
		);
		
		// Check that we have actually fully removed the whole range from the bottom map
		assert(!bottomMap->exists(accessRange, [&](__attribute__((unused)) LinearRegionDataAccessMap::iterator position) -> bool { return true; }));
		
		// Insert the new DataAccess
		bottomMap->insert(LinearRegionDataAccessMapNode(accessRange, dataAccess));
		task->getDataAccesses().push_back(*dataAccess);
		
		if (dataAccess->_blockerCount == 0) {
			Instrument::dataAccessBecomesSatisfied(
				dataAccess->_instrumentationId,
				false, false, true,
				task->getInstrumentationTaskId(),
				task->getInstrumentationTaskId()
			);
		} else if (!weak) {
			task->increasePredecessors();
		}
	}
	
	
public:

	//! \brief adds a task access taking into account repeated accesses
	//! 
	//! \param[inout] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] weak true iff the access is weak
	//! \param[in] accessRange the range of data covered by the access
	//! \param[in] superAccess the access of the parent that contains the new access or nullptr if there is none
	//! \param[in] lock a pointer to the lock that protects the hierarchy of accesses that leads to the new access
	//! \param[inout] topMap the map that contains the information to calculate input dependencies
	//! \param[inout] bottomMap the map that contains the information to calculate output dependencies
	static inline void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange accessRange,
		DataAccess *superAccess, SpinLock *lock,
		LinearRegionDataAccessMap *topMap, LinearRegionDataAccessMap *bottomMap
	) {
		assert(task != nullptr);
		assert(lock != nullptr);
		assert(bottomMap != nullptr);
		
		std::unique_lock<SpinLock> guard(*lock);
		
		registerTaskDataAccessPrelocked(task, accessType, weak, accessRange, superAccess, lock, topMap, bottomMap);
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
		CPU *cpu = currentThread->getComputePlace();
		assert(cpu != 0);
		
		// A temporary list of tasks to minimize the time spent with the mutex held.
		CPUDependencyData::satisfied_originator_list_t &satisfiedOriginators = cpu->_dependencyData._satisfiedAccessOriginators;
		
		TaskDataAccesses &taskDataAccesses = finishedTask->getDataAccesses();
		for (auto it = taskDataAccesses.begin(); it != taskDataAccesses.end(); ) {
			DataAccess *dataAccess = &(*it);
			
			assert(dataAccess->_originator == finishedTask);
			{
				// Locking strategy:
				// 	Every DataAccess that accesses the same data is protected by the same SpinLock that is located
				// 	at the root LockedLinearRegionMap of the hierachy of accesses to the same data. Every DataAccess
				// 	of the same hierarchy points to that SpinLock.
				std::unique_lock<SpinLock> guard(*dataAccess->_lock);
				
				LinearRegionDataAccessMap *topMap = nullptr;
				LinearRegionDataAccessMap *bottomMap = dataAccess->_bottomMap;
				
				if (dataAccess->_superAccess != nullptr) {
					topMap = &dataAccess->_superAccess->_topSubaccesses;
				}
				
				unregisterDataAccess(
					finishedTask->getInstrumentationTaskId(),
					dataAccess, topMap, bottomMap,
					/* OUT */ satisfiedOriginators
				);
			}
			processSatisfiedOriginators(satisfiedOriginators, cpu);
			satisfiedOriginators.clear();
			
			it = taskDataAccesses.erase(it);
			delete dataAccess;
		}
	}
	
	
	static inline void handleEnterBlocking(__attribute__((unused)) Task *task)
	{
	}
	static inline void handleExitBlocking(__attribute__((unused)) Task *task)
	{
	}
	
	static inline void handleEnterTaskwait(__attribute__((unused)) Task *task)
	{
	}
	static inline void handleExitTaskwait(__attribute__((unused)) Task *task)
	{
	}
	
};


#endif // DATA_ACCESS_REGISTRATION_HPP

