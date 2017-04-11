#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

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
private:
	
	static inline DataAccess *createAccess(Task *originator, DataAccessType accessType, bool weak, DataAccessRange range)
	{
		Instrument::data_access_id_t newDataAccessInstrumentationId;
		
		// Regular object duplication
		DataAccess *dataAccess = new DataAccess(
			accessType, weak, originator, range,
			newDataAccessInstrumentationId
		);
		
		return dataAccess;
	}
	
	
	static inline DataAccess *createAccess(Task *originator, DataAccessType accessType, bool weak, DataAccessRange range, int reductionOperation)
	{
		DataAccess *dataAccess = createAccess(originator, accessType, weak, range);
		dataAccess->_reductionInfo = reductionOperation;
		
		return dataAccess;
	}
	
	
	static inline void upgradeAccess(DataAccess *dataAccess, DataAccessType accessType, bool weak)
	{
		assert(dataAccess != nullptr);
		assert(!dataAccess->hasBeenDiscounted());
		
		FatalErrorHandler::failIf(
				(dataAccess->_type == CONCURRENT_ACCESS_TYPE) || (accessType == CONCURRENT_ACCESS_TYPE),
				"when registering accesses for task ",
				(dataAccess->_originator->getTaskInfo()->task_label != nullptr ?
					dataAccess->_originator->getTaskInfo()->task_label :
					dataAccess->_originator->getTaskInfo()->declaration_source),
				": Combining accesses of type concurrent with other accesses over the same data is not permitted");
		
		FatalErrorHandler::failIf(
				(dataAccess->_type == REDUCTION_ACCESS_TYPE) || (accessType == REDUCTION_ACCESS_TYPE),
				"when registering accesses for task ",
				(dataAccess->_originator->getTaskInfo()->task_label != nullptr ?
					dataAccess->_originator->getTaskInfo()->task_label :
					dataAccess->_originator->getTaskInfo()->declaration_source),
				": Combining reduction accesses with other accesses over the same data is not permitted");
		
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
			toBeDuplicated._type,
			toBeDuplicated._weak,
			toBeDuplicated._range,
			toBeDuplicated._reductionInfo
		);
		
		newFragment->_status = toBeDuplicated._status;
		newFragment->_next = toBeDuplicated._next;
		newFragment->_child = toBeDuplicated._child;
		
		if (updateTaskBlockingCount && !newFragment->_weak && !newFragment->satisfied()) {
			toBeDuplicated._originator->increasePredecessors();
		}
		
		assert(accessStructures._lock.isLockedByThisThread() || noAccessIsReachable(accessStructures));
		
		return newFragment;
	}
	
	static inline void removeDataAccess(
		Task *task, DataAccess *dataAccess,
		CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(dataAccess != nullptr);
		
		assert(dataAccess->complete());
		assert(dataAccess->readSatisfied());
		assert(dataAccess->writeSatisfied());
		
		DataAccessRange range = dataAccess->_range;
		assert(!range.empty());
		
		if (dataAccess->isInBottomMap()) {
			CPUDependencyData::data_access_range_list_t &removedRangeList =
				cpuDependencyData._removedRangesFromBottomMap;
			
			if (removedRangeList.empty()) {
				removedRangeList.push_back(range);
			} else {
				DataAccessRange lastRange(removedRangeList.back());
				assert(range.intersect(lastRange).empty());
				
				if (lastRange.contiguous(range)) {
					removedRangeList.pop_back();
					removedRangeList.push_back(lastRange.contiguousUnion(range));
				} else {
					removedRangeList.push_back(range);
				}
			}
		}
		
		if (dataAccess->_next != nullptr) {
			Instrument::unlinkedDataAccesses(
				dataAccess->_instrumentationId,
				dataAccess->_next->getInstrumentationTaskId(),
				/* direct */ true
			);
		}
		
		Instrument::removedDataAccess(
			dataAccess->_instrumentationId
		);
		
		delete dataAccess;
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
				} else {
					Instrument::modifiedDataAccessRange(fragment->_instrumentationId, fragment->_range);
				}
			}
		);
		
		dataAccess = &(*position);
		assert(dataAccess != nullptr);
		assert(dataAccess->_range.fullyContainedIn(range));
		
		return dataAccess;
	}
	
	
	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
		/* INOUT */ CPUDependencyData &cpuDependencyData,
		ComputePlace *computePlace
	) {
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : cpuDependencyData._satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			ComputePlace *idleComputePlace = Scheduler::addReadyTask(satisfiedOriginator, computePlace, SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT);
			
			if (idleComputePlace != nullptr) {
				ThreadManager::resumeIdle((CPU *) idleComputePlace);
			}
		}
		
		cpuDependencyData._satisfiedOriginators.clear();
	}
	
	
	static void processRemovableTasks(
		/* inout */ CPUDependencyData &cpuDependencyData,
		ComputePlace *computePlace
	) {
		assert(computePlace != nullptr);
		
		for (Task *removableTask : cpuDependencyData._removableTasks) {
			assert(removableTask != nullptr);
			
			TaskFinalization::disposeOrUnblockTask(removableTask, computePlace);
		}
		cpuDependencyData._removableTasks.clear();
	}
	
	
	static inline void processRemovedRangesFromBottomMap(
		Task *task, Task *parent,
		TaskDataAccesses &parentAccessStructures,
		CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(parent != nullptr);
		assert(!parentAccessStructures.hasBeenDeleted());
		
		for (DataAccessRange range : cpuDependencyData._removedRangesFromBottomMap) {
			assert(!range.empty());
			
			parentAccessStructures._accesses.processIntersecting(
				range,
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *parentAccess = &(*position);
					assert(parentAccess != nullptr);
					assert(parentAccess->_child != nullptr);
					assert(parentAccess->hasSubaccesses());
					
					if (!parentAccess->_range.fullyContainedIn(range)) {
						parentAccess = fragmentAccess(
							parentAccess, range,
							parentAccessStructures,
							/* Consider blocking */ true
						);
						assert(parentAccess != nullptr);
					}
					
					parentAccess->_child = nullptr;
					parentAccess->hasSubaccesses() = false;
					
					return true;
				}
			);
			
			parentAccessStructures._subaccessBottomMap.processIntersecting(
				range,
				[&](TaskDataAccesses::subaccess_bottom_map_t::iterator position) -> bool {
					BottomMapEntry *bottomMapEntry = &(*position);
					assert(bottomMapEntry != nullptr);
					assert(bottomMapEntry->_task == task);
					
					if (!bottomMapEntry->_range.fullyContainedIn(range)) {
						bottomMapEntry = fragmentBottomMapEntry(
							bottomMapEntry, range,
							parentAccessStructures
						);
						assert(bottomMapEntry != nullptr);
					}
					
					parentAccessStructures._subaccessBottomMap.erase(*bottomMapEntry);
					
					return true;
				}
			);
		}
		
		cpuDependencyData._removedRangesFromBottomMap.clear();
	}
	
	
	template <typename MatchingProcessorType, typename MissingProcessorType>
	static inline bool foreachBottomMapMatchingAndMissingRange(
		Task *parent, TaskDataAccesses &parentAccessStructures,
		DataAccessRange range, __attribute__((unused)) Task *task, TaskDataAccesses &accessStructures,
		MatchingProcessorType matchingProcessor, MissingProcessorType missingProcessor,
		bool removeBottomMapEntry
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert(!range.empty());
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
				assert(subtask != parent);
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				assert(!subtaskAccessStructures.hasBeenDeleted());
				
				if (subtask != parent) {
					// Take care of the possible deadlock
					if (!subtaskAccessStructures._lock.tryLock()) {
						accessStructures._lock.unlock();
						subtaskAccessStructures._lock.lock();
						accessStructures._lock.lock();
					}
				}
				
				// For each access of the subtask that matches
				bool result = subtaskAccessStructures._accesses.processIntersectingWithRecentAdditions(
					subrange,
					[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
						DataAccess *previous = &(*accessPosition);
						
						assert(previous->_next == nullptr);
						assert(previous->isInBottomMap());
						assert(!previous->hasBeenDiscounted());
						
						return matchingProcessor(previous, bottomMapEntry);
					}
				);
				
				if (subtask != parent) {
					subtaskAccessStructures._lock.unlock();
				}
				
				// Remove bottom map entries indexed in the actual parent
				if (removeBottomMapEntry) {
					bottomMapEntry = fragmentBottomMapEntry(bottomMapEntry, range, parentAccessStructures);
					parentAccessStructures._subaccessBottomMap.erase(*bottomMapEntry);
					delete bottomMapEntry;
				}
				
				return result;
			},
			[&](DataAccessRange missingRange) -> bool {
				parentAccessStructures._accesses.processIntersectingAndMissingWithRecentAdditions(
					missingRange,
					[&](TaskDataAccesses::accesses_t::iterator superaccessPosition) -> bool {
						DataAccess *previous = &(*superaccessPosition);
						assert(previous != nullptr);
						
						return matchingProcessor(previous, nullptr);
					},
					[&](DataAccessRange rangeUncoveredByParent) -> bool {
						return missingProcessor(rangeUncoveredByParent);
					}
				);
				
				return true;
			}
		);
	}
	
	
	static inline void propagateSatisfiabilityAfterFirstLinking(
		Task *previous, DataAccess *previousAccess,
		Task *next, DataAccess *nextAccess,
		bool parentalRelation = false
	) {
		assert(previous != nullptr);
		assert(previousAccess != nullptr);
		assert(next != nullptr);
		assert(nextAccess != nullptr);
		
		bool readSatisfiability = false;
		bool writeSatisfiability = false;
		bool topmostSatisfiability = false;
		
		evaluateSatisfiability(
			previous, previousAccess,
			next, nextAccess,
			readSatisfiability,
			writeSatisfiability,
			topmostSatisfiability,
			parentalRelation
		);
		
		nextAccess->readSatisfied() = readSatisfiability;
		nextAccess->writeSatisfied() = writeSatisfiability;
		nextAccess->topmostSatisfied() = topmostSatisfiability;
		if (nextAccess->satisfied() && !nextAccess->_weak) {
			next->decreasePredecessors();
		}
		
		if (topmostSatisfiability) {
			Instrument::dataAccessBecomesRemovable(nextAccess->_instrumentationId);
			
			TaskDataAccesses &nextAccessStructures = next->getDataAccesses();
			nextAccessStructures.decreaseRemovalCount(
				nextAccess->_range.getSize()
			);
		}
		
		Instrument::dataAccessBecomesSatisfied(
			nextAccess->_instrumentationId,
			readSatisfiability, writeSatisfiability, false,
			next->getInstrumentationTaskId()
		);
	}
	
	static inline void propagateSatisfiabilityAfterLinkingBottomMapAccessesToNext(
		DataAccess *dataAccess, Task *task,
		Task *next,
		CPUDependencyData &cpuDependencyData
	) {
		assert(dataAccess != nullptr);
		assert(task != nullptr);
		assert(next != nullptr);
		
		propagateSatisfiability(
			task, dataAccess, 
			dataAccess->_range, next,
			cpuDependencyData
		);
	}
	
	static inline void evaluateSatisfiability(
		Task *previous, DataAccess *previousAccess,
		Task *next, DataAccess *nextAccess,
		bool &readSatisfiability, bool &writeSatisfiability,
		bool &topmostSatisfiability,
		bool parentalRelation = false
	) {
		assert(previous != nullptr);
		assert(previousAccess != nullptr);
		assert(next != nullptr);
		assert(nextAccess != nullptr);
		
		readSatisfiability =
			previousAccess->readSatisfied() && (
					// Previous is a complete, non-(concurrent/reduction) access or concurrent topmost access
					(previousAccess->complete() &&
						((previousAccess->_type != CONCURRENT_ACCESS_TYPE &&
							previousAccess->_type != REDUCTION_ACCESS_TYPE) ||
						previousAccess->topmostSatisfied())) ||
					// There is a paternal relation between accesses
					parentalRelation ||
					// Previous is a read access
					previousAccess->_type == READ_ACCESS_TYPE ||
					// Both are concurrent accesses
					(previousAccess->_type == CONCURRENT_ACCESS_TYPE &&
						nextAccess->_type == CONCURRENT_ACCESS_TYPE) ||
					// Both are same-type reduction accesses
					(previousAccess->_type == REDUCTION_ACCESS_TYPE &&
						nextAccess->_type == REDUCTION_ACCESS_TYPE &&
						previousAccess->_reductionInfo == nextAccess->_reductionInfo));
		
		writeSatisfiability =
			previousAccess->writeSatisfied() && (
					// Previous is a complete, non-(concurrent/reduction) access or concurrent topmost access
					(previousAccess->complete() &&
						((previousAccess->_type != CONCURRENT_ACCESS_TYPE &&
							previousAccess->_type != REDUCTION_ACCESS_TYPE) ||
						previousAccess->topmostSatisfied())) ||
					// There is a paternal relation between accesses
					parentalRelation ||
					// Both are concurrent accesses
					(previousAccess->_type == CONCURRENT_ACCESS_TYPE &&
						nextAccess->_type == CONCURRENT_ACCESS_TYPE) ||
					// Both are same-type reduction accesses
					(previousAccess->_type == REDUCTION_ACCESS_TYPE &&
						nextAccess->_type == REDUCTION_ACCESS_TYPE &&
						previousAccess->_reductionInfo == nextAccess->_reductionInfo));
		
		topmostSatisfiability =
			previousAccess->topmostSatisfied()
			&& (previousAccess->complete() || parentalRelation);
	}
	
	static inline void propagateSatisfiability(
		Task *task, DataAccess *dataAccess,
		DataAccessRange range, Task *nextTask,
		CPUDependencyData &cpuDependencyData,
		bool parentalRelation = false
	) {
		assert(task != nullptr);
		assert(nextTask != nullptr);
		assert(!range.empty());
		
		TaskDataAccesses &nextAccessStructures = nextTask->getDataAccesses();
		assert(!nextAccessStructures.hasBeenDeleted());
		nextAccessStructures._lock.lock();
		
		nextAccessStructures._accesses.processIntersecting(
			range,
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess *nextAccess = &(*position);
				assert(nextAccess != nullptr);
				
				bool wasSatisfied = nextAccess->satisfied();
				bool wasReadSatisfied = nextAccess->readSatisfied();
				bool wasWriteSatisfied = nextAccess->writeSatisfied();
				bool wasTopmostSatisfied = nextAccess->topmostSatisfied();
				bool readSatisfiability = false;
				bool writeSatisfiability = false;
				bool topmostSatisfiability = false;
				
				evaluateSatisfiability(
					task, dataAccess,
					nextTask, nextAccess,
					readSatisfiability,
					writeSatisfiability,
					topmostSatisfiability,
					parentalRelation
				);
				
				if (wasReadSatisfied && readSatisfiability) {
					readSatisfiability = false;
				}
				if (wasWriteSatisfied && writeSatisfiability) {
					writeSatisfiability = false;
				}
				assert(!(topmostSatisfiability && wasTopmostSatisfied));
				
				if (!readSatisfiability && !writeSatisfiability && !topmostSatisfiability) {
					return true;
				}
				
				assert(readSatisfiability || writeSatisfiability || topmostSatisfiability);
				assert(!(readSatisfiability && wasReadSatisfied));
				assert(!(readSatisfiability && wasWriteSatisfied));
				assert(!(writeSatisfiability && wasWriteSatisfied));
				assert(!(topmostSatisfiability && wasTopmostSatisfied));
				
				// Fragment next access if needed
				if (!nextAccess->_range.fullyContainedIn(range)) {
					nextAccess = fragmentAccess(
						nextAccess, range.intersect(nextAccess->_range),
						nextAccessStructures,
						/* Consider blocking */ true
					);
					assert(nextAccess != nullptr);
				}
				
				// Check some assumptions
				assert(!(nextAccess->complete()
					&& nextAccess->hasSubaccesses()
					&& nextAccess->_next != nullptr)
				);
				
				// Modify the satisfiability of the access
				nextAccess->readSatisfied() = (readSatisfiability) ? true : wasReadSatisfied;
				nextAccess->writeSatisfied() = (writeSatisfiability) ? true : wasWriteSatisfied;
				nextAccess->topmostSatisfied() = topmostSatisfiability;
				
				assert(!(nextAccess->topmostSatisfied() && !nextAccess->readSatisfied()));
				assert(!(nextAccess->topmostSatisfied() && !nextAccess->writeSatisfied()));
				
				Instrument::dataAccessBecomesSatisfied(
					nextAccess->_instrumentationId,
					readSatisfiability, writeSatisfiability, false,
					nextTask->getInstrumentationTaskId()
				);
				
				// Propagate to subaccesses
				if (nextAccess->hasSubaccesses()) {
					propagateSatisfiability(
						nextTask, nextAccess,
						nextAccess->_range,
						nextAccess->_child,
						cpuDependencyData,
						true
					);
				}
				
				if (topmostSatisfiability) {
					Instrument::dataAccessBecomesRemovable(nextAccess->_instrumentationId);
					
					size_t bytes = nextAccess->_range.getSize();
					if (nextAccessStructures.decreaseRemovalCount(bytes)) {
						if (nextTask->decreaseRemovalBlockingCount()) {
							cpuDependencyData._removableTasks.push_back(nextTask);
						}
					}
				}
				
				// Add the task to the satisfied originators if it is ready
				if (!wasSatisfied && nextAccess->satisfied() && !nextAccess->_weak) {
					if (nextTask->decreasePredecessors()) {
						cpuDependencyData._satisfiedOriginators.push_back(nextTask);
					}
				}
				
				// Propagate to next task
				if (nextAccess->_next != nullptr) {
					propagateSatisfiability(
						nextTask, nextAccess,
						nextAccess->_range,
						nextAccess->_next,
						cpuDependencyData
					);
				}
				
				return true;
			}
		);
		
		nextAccessStructures._lock.unlock();
	}
	
	
	static inline DataAccess *linkAndPropagateAfterFirstLinking(
		DataAccess *dataAccess, Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range, DataAccess *nextDataAccess, Task *next,
		TaskDataAccesses &nextAccessStructures
	) {
		assert(dataAccess != nullptr);
		assert(dataAccess->isReachable());
		assert(!dataAccess->hasBeenDiscounted());
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(nextDataAccess != nullptr);
		assert(next != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		
		Task *taskParent = task->getParent();
		Task *nextParent = next->getParent();
		assert(taskParent != nullptr);
		assert(nextParent != nullptr);
		
		bool parentalRelation = (task == nextParent);
		bool siblingRelation = (taskParent == nextParent);
		assert(dataAccess->isInBottomMap() || parentalRelation);
		assert(dataAccess->_next == nullptr || parentalRelation);
		
		dataAccess = fragmentAccess(
			dataAccess, range,
			accessStructures,
			/* Consider blocking */ true
		);
		assert(dataAccess != nullptr);
		assert(dataAccess->isInBottomMap() || parentalRelation);
		assert(dataAccess->_next == nullptr || parentalRelation);
		
		nextDataAccess = fragmentAccess(
			nextDataAccess, range,
			nextAccessStructures,
			/* Consider blocking */ true
		);
		assert(nextDataAccess != nullptr);
		assert(nextDataAccess->_next == nullptr);
		
		// Link the dataAccess access to the new task
		if (parentalRelation) {
			dataAccess->_child = next;
			dataAccess->hasSubaccesses() = true;
		} else {
			dataAccess->_next = next;
			if (siblingRelation) {
				dataAccess->isInBottomMap() = false;
			}
		}
		
		Instrument::linkedDataAccesses(
			dataAccess->_instrumentationId, next->getInstrumentationTaskId(),
			dataAccess->_range,
			true, false
		);
		
		propagateSatisfiabilityAfterFirstLinking(
			task, dataAccess,
			next, nextDataAccess,
			parentalRelation
		);
		
		// Return the data access since it may have been fragmented
		return dataAccess;
	}
	
	
	static inline void linkBottomMapAccessesToNext(
		Task *task, TaskDataAccesses &accessStructures,
		DataAccessRange range, Task *next,
		CPUDependencyData &cpuDependencyData
	) {
		assert(task != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures._lock.isLockedByThisThread());
		assert(!range.empty());
		assert(next != nullptr);
		
		accessStructures._subaccessBottomMap.processIntersecting(
			range,
			[&](TaskDataAccesses::subaccess_bottom_map_t::iterator bottomMapPosition) -> bool {
				BottomMapEntry *bottomMapEntry = &(*bottomMapPosition);
				assert(bottomMapEntry != nullptr);
				
				Task *subtask = bottomMapEntry->_task;
				assert(subtask != nullptr);
				assert(subtask != task);
				
				DataAccessRange subrange = range.intersect(bottomMapEntry->_range);
				assert(!subrange.empty());
				
				TaskDataAccesses &subtaskAccessStructures = subtask->getDataAccesses();
				subtaskAccessStructures._lock.lock();
				
				// For each access of the subtask that matches
				subtaskAccessStructures._accesses.processIntersectingWithRecentAdditions(
					subrange,
					[&] (TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
						DataAccess *subaccess = &(*accessPosition);
						assert(subaccess != nullptr);
						assert(subaccess->isReachable());
						assert(subaccess->_next == nullptr);
						assert(subaccess->isInBottomMap());
						assert(!subaccess->hasBeenDiscounted());
						
						DataAccessRange rangeToBeProcessed = subrange.intersect(subaccess->_range);
						assert(!rangeToBeProcessed.empty());
						
						if (!subaccess->hasSubaccesses() || !subaccess->complete()) {
							// Fragment the subaccess if needed
							subaccess = fragmentAccess(
								subaccess, rangeToBeProcessed,
								subtaskAccessStructures,
								/* Affect originator blocking counter */ true
							);
							assert(subaccess != nullptr);
							
							// Assign the next task
							subaccess->_next = next;
							
							Instrument::linkedDataAccesses(
								subaccess->_instrumentationId,
								next->getInstrumentationTaskId(),
								rangeToBeProcessed, true, false
							);
							
							// Propagate the satisfiability to the recently linked access
							propagateSatisfiabilityAfterLinkingBottomMapAccessesToNext(
								subaccess, subtask,
								next,
								cpuDependencyData
							);
						} else {
							linkBottomMapAccessesToNext(
								subtask, subtaskAccessStructures,
								rangeToBeProcessed, next,
								cpuDependencyData
							);
						}
						
						return true;
					}
				);
				
				subtaskAccessStructures._lock.unlock();
				
				return true;
			}
		);
	}
	
	
	static inline void replaceAndLinkToNotCompletedOrWithoutSubaccessPredecessors(
		Task *task, TaskDataAccesses &accessStructures, DataAccess *dataAccess,
		Task *parent, TaskDataAccesses &parentAccessStructures
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert(dataAccess != nullptr);
		assert(!accessStructures.hasBeenDeleted());
		assert(!parentAccessStructures.hasBeenDeleted());
		
		DataAccessRange range = dataAccess->_range;
		assert(!range.empty());
		
		// The satisfiability propagation will decrease the predecessor count as needed
		if (!dataAccess->_weak) {
			task->increasePredecessors();
		}
		
		bool local = false;

		linkToNotCompletedOrWithoutSubaccessesPredecessors(
			task, accessStructures,
			dataAccess, range,
			parent, parentAccessStructures,
			local
		);
		
		// Add the entry to the bottom map
		BottomMapEntry *bottomMapEntry = new BottomMapEntry(range, task, local);
		parentAccessStructures._subaccessBottomMap.insert(*bottomMapEntry);
	}
	
	
	static inline void linkToNotCompletedOrWithoutSubaccessesPredecessors(
		Task *task, TaskDataAccesses &accessStructures,
		DataAccess *dataAccess, DataAccessRange range,
		Task *parent, TaskDataAccesses &parentAccessStructures,
		bool &local
	) {
		assert(parent != nullptr);
		assert(task != nullptr);
		assert(dataAccess != nullptr);
		assert(range.fullyContainedIn(dataAccess->_range));
		assert(!accessStructures.hasBeenDeleted());
		assert(!parentAccessStructures.hasBeenDeleted());
		
		// Get the iterator to the linking data access
		TaskDataAccesses::accesses_t::iterator partialDataAccessPosition = accessStructures._accesses.iterator_to(*dataAccess);
		assert(partialDataAccessPosition != accessStructures._accesses.end());
		
		// Determine if they are actually parent and son
		bool parentalRelation = (task->getParent() == parent);
		
		// Link accesses to their corresponding predecessor
		foreachBottomMapMatchingAndMissingRange(
			parent, parentAccessStructures,
			range, task, accessStructures,
			[&](DataAccess *previous, BottomMapEntry *bottomMapEntry) -> bool {
				assert(previous != nullptr);
				assert(previous->isReachable());
				assert(!previous->hasBeenDiscounted());
				assert(!range.intersect(previous->_range).empty());
				assert(partialDataAccessPosition != accessStructures._accesses.end());
				
				DataAccess *partialDataAccess = &(*partialDataAccessPosition);
				assert(partialDataAccess != nullptr);
				assert(!partialDataAccess->hasBeenDiscounted());
				
				DataAccessRange rangeToBeProcessed = range.intersect(previous->_range);
				assert(!rangeToBeProcessed.empty());
				
				Task *previousTask = previous->_originator;
				assert(previousTask != nullptr);
				assert(bottomMapEntry == nullptr || previousTask == bottomMapEntry->_task);
				
				TaskDataAccesses &previousAccessStructures = previousTask->getDataAccesses();
				assert(!previousAccessStructures.hasBeenDeleted());
				
				// Can be modified only from the same domain
				if (parentalRelation) {
					if (bottomMapEntry != nullptr) {
						local = bottomMapEntry->_local;
					} else {
						// The first subaccess of a parent access
						local = false;
					}
				}
				
				// In case previous task is not the parent, continue recursively
				// if previous is completed and has subaccesses
				if (previousTask != task->getParent() && previous->hasSubaccesses() && previous->complete()) {
					// Avoid reprocessing some new additions
					++partialDataAccessPosition;
					
					// Explore and link it to previous's subaccesses
					linkToNotCompletedOrWithoutSubaccessesPredecessors(
        				task, accessStructures,
        				partialDataAccess, rangeToBeProcessed,
        				previousTask, previousAccessStructures,
						local
					);
					
					if (parentalRelation) {
						previous = fragmentAccess(
							previous, rangeToBeProcessed,
							previousAccessStructures,
							/* Consider blocking */ true
						);
						assert(previous != nullptr);
						assert(previous->_range.fullyContainedIn(range));
						
						previous->isInBottomMap() = false;
					}
					
					// Go back to the next partial access
					--partialDataAccessPosition;
				}
				else {
					previous = linkAndPropagateAfterFirstLinking(
						previous, previousTask, previousAccessStructures,
						rangeToBeProcessed, partialDataAccess,
						task, accessStructures
					);
					
					// Advance to the next partial access
					++partialDataAccessPosition;
				}
				
				return true;
			},
			[&](DataAccessRange missingRange) -> bool {
				assert(parentalRelation);
				assert(!parentAccessStructures._accesses.contains(missingRange));
				assert(partialDataAccessPosition != accessStructures._accesses.end());
				
				// Not part of the parent
				local = true;
				
				DataAccess *partialDataAccess = &(*partialDataAccessPosition);
				assert(partialDataAccess != nullptr);
				assert(partialDataAccess->_range.fullyContainedIn(range));
				assert(!partialDataAccess->hasBeenDiscounted());
				
				partialDataAccess = fragmentAccess(
					partialDataAccess, missingRange,
					accessStructures,
					/* Consider blocking */ true
				);
				
				partialDataAccess->readSatisfied() = true;
				partialDataAccess->writeSatisfied() = true;
				partialDataAccess->topmostSatisfied() = true;
				if (!partialDataAccess->_weak) {
					task->decreasePredecessors();
				}
				
				Instrument::dataAccessBecomesRemovable(
					partialDataAccess->_instrumentationId
				);
				
				accessStructures.decreaseRemovalCount(
					partialDataAccess->_range.getSize()
				);
				
				Instrument::dataAccessBecomesSatisfied(
					partialDataAccess->_instrumentationId,
					true, true, false,
					task->getInstrumentationTaskId()
				);

				// Advance to the next partial access
				++partialDataAccessPosition;
				
				return true;
			},
			parentalRelation
		);
	}
	
	
	static inline void linkTaskAccesses(
		Task *task
	) {
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		if (accessStructures._accesses.empty()) {
			return;
		}
		
		// It will be decreased when unregistering accesses
		accessStructures.increaseRemovalCount();
		
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
					
					dataAccess->isInBottomMap() = true;
					
#ifndef NDEBUG
					dataAccess->isReachable() = true;
#endif
					
					replaceAndLinkToNotCompletedOrWithoutSubaccessPredecessors(
						task, accessStructures, dataAccess,
						parent, parentAccessStructures
					);
					
					return true;
				}
			);
		}
	}
	
	
	static inline void finalizeAccess(
		Task *finishedTask, DataAccess *dataAccess,
		CPUDependencyData &cpuDependencyData
	) {
		assert(finishedTask != nullptr);
		assert(dataAccess != nullptr);
		assert(dataAccess->_originator == finishedTask);
		
		// The access may already have been released through the "release" directive
		if (dataAccess->complete()) {
			return;
		}
		assert(!dataAccess->hasBeenDiscounted());
		
		TaskDataAccesses &accessStructures = finishedTask->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		DataAccessRange range = dataAccess->_range;
		assert(!range.empty());
		
		// Mark it as complete
		Instrument::completedDataAccess(dataAccess->_instrumentationId);
		assert(!dataAccess->complete());
		dataAccess->complete() = true;
		
		if (dataAccess->hasSubaccesses() && dataAccess->_next != nullptr) {
			// Link bottom map subaccesses to the next
			linkBottomMapAccessesToNext(
				dataAccess->_originator, accessStructures, range,
				dataAccess->_next,
				cpuDependencyData
			);
			
			Instrument::unlinkedDataAccesses(
				dataAccess->_instrumentationId,
				dataAccess->_next->getInstrumentationTaskId(),
				/* direct */ true
			);
			
			dataAccess->_next = nullptr;
		}
		
		assert(!dataAccess->hasSubaccesses() || dataAccess->_next == nullptr);
		
		if (dataAccess->_next != nullptr) {
			bool readSatisfied = dataAccess->readSatisfied();
			bool writeSatisfied = dataAccess->writeSatisfied();
			
			if (readSatisfied || writeSatisfied) {
				propagateSatisfiability(
					finishedTask, dataAccess,
					range, dataAccess->_next,
					cpuDependencyData
				);
			}
		}
	}
	
	
public:

	//! \brief creates a task data access taking into account repeated accesses but does not link it to previous accesses nor superaccesses
	//! 
	//! \param[in,out] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] weak true iff the access is weak
	//! \param[in] range the range of data covered by the access
	template<typename... ReductionInfo>
	static inline void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRange range, ReductionInfo... reductionInfo
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
				DataAccess *newAccess = createAccess(task, accessType, weak, missingRange, reductionInfo...);
				
				accessStructures.increaseRemovalCount(missingRange.getSize());
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
	static inline bool registerTaskDataAccesses(Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
		assert(task != 0);
		assert(task != nullptr);
		assert(computePlace != nullptr);
		
		nanos_task_info *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		
		// This part creates the DataAccesses and calculates any possible upgrade
		taskInfo->register_depinfo(task, task->getArgsBlock());
		
		if (!task->getDataAccesses()._accesses.empty()) {
			// The blocking count is decreased once all the accesses become removable
			task->increaseRemovalBlockingCount();
			
			task->increasePredecessors(2);
			
			// This part actually inserts the accesses into the dependency system
			linkTaskAccesses(task);
			
			return task->decreasePredecessors(2);
		} else {
			return true;
		}
	}
	
	
	static inline void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace)
	{
		assert(task != 0);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		
		if (accesses.empty()) {
			return;
		}
		
		CPUDependencyData &cpuDependencyData = computePlace->getDependencyData();
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		
		Task *parent = task->getParent();
		assert(parent != nullptr);
		
		TaskDataAccesses &parentAccessStructures = parent->getDataAccesses();
		assert(!parentAccessStructures.hasBeenDeleted());
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> parentGuard(parentAccessStructures._lock);
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.processAll(
				[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *dataAccess = &(*position);
					assert(dataAccess != nullptr);
					
					finalizeAccess(task, dataAccess, cpuDependencyData);
					
					return true;
				}
			);
			
			if (accessStructures.decreaseRemovalCount()) {
				task->decreaseRemovalBlockingCount();
			}
		}
		
		// Schedule satisfied tasks
		processSatisfiedOriginators(cpuDependencyData, computePlace);
		
		// Recycle removable tasks
		processRemovableTasks(cpuDependencyData, computePlace);
		
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
		if (!accessStructures.isRemovable()) {
			task->decreaseRemovalBlockingCount();
		}
	}
	
	
	static void handleExitBlocking(Task *task)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		if (!accessStructures.isRemovable()) {
			task->increaseRemovalBlockingCount();
		}
	}
	
	
	static void handleEnterTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		if (!accessStructures.isRemovable()) {
			task->decreaseRemovalBlockingCount();
		}
	}
	
	
	static void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		
		// In principle, all inner tasks must have ended
		assert(accessStructures._subaccessBottomMap.empty());
		
		if (!accessStructures.isRemovable()) {
			task->increaseRemovalBlockingCount();
		}
	}
	
	
	static void handleTaskRemoval(Task *task, ComputePlace *computePlace)
	{
		assert(task != 0);
		assert(task != nullptr);
		assert(computePlace != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		assert(accessStructures.isRemovable());
		
		TaskDataAccesses::accesses_t &accesses = accessStructures._accesses;
		assert(accessStructures._subaccessBottomMap.empty());
		
		if (accesses.empty()) {
			return;
		}
		
		Task *parent = task->getParent();
		assert(parent != nullptr);
		
		TaskDataAccesses &parentAccessStructures = parent->getDataAccesses();
		assert(!parentAccessStructures.hasBeenDeleted());
		
		CPUDependencyData &cpuDependencyData = computePlace->getDependencyData();
		
#ifndef NDEBUG
		bool alreadyTaken = false;
		cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, true);
#endif
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> parentGuard(parentAccessStructures._lock);
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
			
			accesses.deleteAll(
				[&](DataAccess *access) {
					assert(access != nullptr);
					
					removeDataAccess(
						task, access,
						cpuDependencyData
					);
				}
			);
			assert(accesses.empty());
			
			// NOTE: mutexes must be held
			processRemovedRangesFromBottomMap(
				task, parent,
				parentAccessStructures,
				cpuDependencyData
			);
		}
		
#ifndef NDEBUG
		if (!alreadyTaken) {
			bool alreadyTaken = true;
			assert(cpuDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif

	}
};


#endif // DATA_ACCESS_REGISTRATION_HPP
