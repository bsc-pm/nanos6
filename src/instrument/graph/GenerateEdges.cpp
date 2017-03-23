#include "GenerateEdges.hpp"
#include "InstrumentGraph.hpp"

#include <cassert>


namespace Instrument {
	namespace Graph {
		
		static void markAccessesWithPrevious()
		{
			for (auto &idAndAccess : _accessIdToAccessMap) {
				access_t *access = idAndAccess.second;
				assert(access != nullptr);
				
				if (access->fragment()) {
					continue;
				}
				
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						nextAccess.hasPrevious() = true;
						return true;
					}
				);
			}
			
		}
		
		
		static void buildPredecessorList(
			DataAccessRange range,
			predecessors_t &lastReaders, predecessors_t &lastWriters, predecessors_t &newWriters,
			link_to_next_t &link, access_t *access, task_info_t &taskInfo
		) {
			assert(access != nullptr);
// 			assert(access->hasPrevious());
			assert(range.fullyContainedIn(access->_accessRange));
			
			
			// Advance through the fragments
			for (phase_t *phase : taskInfo._phaseList) {
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(phase);
				
				if (taskGroup == nullptr) {
					continue;
				}
				
				taskGroup->_liveFragments.processIntersecting(
					range,
					[&](task_live_access_fragments_t::iterator position) -> bool {
						access_t *fragment = position->_fragment;
						assert(fragment != nullptr);
						
						DataAccessRange subrange = fragment->_accessRange.intersect(range);
						assert(!subrange.empty());
						
						foreachLiveNextOfAccess(fragment,
							[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
								DataAccessRange nextRange = subrange.intersect(nextAccess._accessRange);
								
								if (!nextRange.empty()) {
									buildPredecessorList(
										nextRange,
										lastReaders, lastWriters, newWriters,
										linkToNext, &nextAccess, nextTaskInfo
									);
								}
								
								return true;
							}
						);
						
						return true;
					}
				);
			}
			
			task_id_t parentId = taskInfo._parent;
			
			if (access->_type == READ_ACCESS_TYPE) {
				assert((!lastReaders.empty() || !lastWriters.empty()) && newWriters.empty());
				
				// Add previous writers as predecessors
				for (task_id_t writer : lastWriters) {
					link._predecessors.insert(writer);
					
					task_info_t &writerTaskInfo = _taskToInfoMap[writer];
					if ((writerTaskInfo._parent == parentId) && (writerTaskInfo._taskGroupPhaseIndex == taskInfo._taskGroupPhaseIndex)) {
						taskInfo._hasPredecessorsInSameLevel = true;
						writerTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				// Insert self as reader
				auto positionAndConfirmation = lastReaders.insert(access->_originator);
				assert(positionAndConfirmation.second);
				
				// Advance to the next access
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						DataAccessRange nextRange = range.intersect(nextAccess._accessRange);
						
						if (!nextRange.empty()) {
							predecessors_t nextNewWriters;
							// Next access hasn't type read so we flush writers
							if (nextAccess._type != READ_ACCESS_TYPE) {
								predecessors_t nextLastWriters;
								
								buildPredecessorList(
									nextRange,
									lastReaders, nextLastWriters, nextNewWriters,
									linkToNext, &nextAccess, nextTaskInfo
								);
							} else {
								buildPredecessorList(
									nextRange,
									lastReaders, lastWriters, nextNewWriters,
									linkToNext, &nextAccess, nextTaskInfo
								);
							}
						}
						
						return true;
					}
				);
				
				// Remove self as reader to return
				lastReaders.erase(positionAndConfirmation.first);
				
			} else if (access->_type == CONCURRENT_ACCESS_TYPE) {
				// Either there are previous readers or writers, but not both
				assert((lastReaders.empty() != lastWriters.empty()) || !newWriters.empty());
				
				// Add previous readers as predecessors
				for (task_id_t reader : lastReaders) {
					link._predecessors.insert(reader);
					
					task_info_t &readerTaskInfo = _taskToInfoMap[reader];
					if ((readerTaskInfo._parent == parentId) && (readerTaskInfo._taskGroupPhaseIndex == taskInfo._taskGroupPhaseIndex)) {
						taskInfo._hasPredecessorsInSameLevel = true;
						readerTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				// Add previous writers as predecessors
				for (task_id_t writer : lastWriters) {
					link._predecessors.insert(writer);
				
					task_info_t &writerTaskInfo = _taskToInfoMap[writer];
					if ((writerTaskInfo._parent == parentId) && (writerTaskInfo._taskGroupPhaseIndex == taskInfo._taskGroupPhaseIndex)) {
						taskInfo._hasPredecessorsInSameLevel = true;
						writerTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				// Insert self as writer
				auto positionAndConfirmation = newWriters.insert(access->_originator);
				assert(positionAndConfirmation.second);
				
				// Advance to the next access
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						DataAccessRange nextRange = range.intersect(nextAccess._accessRange);
						
						if (!nextRange.empty()) {
							// Next access hasn't type concurrent so we flush readers and writers
							if (nextAccess._type != CONCURRENT_ACCESS_TYPE) {
								predecessors_t nextLastReaders;
								predecessors_t nextNewWriters;
								predecessors_t& nextLastWriters = newWriters;
								
								buildPredecessorList(
									nextRange,
									nextLastReaders, nextLastWriters, nextNewWriters,
									linkToNext, &nextAccess, nextTaskInfo
								);
							} else {
								buildPredecessorList(
									nextRange,
									lastReaders, lastWriters, newWriters,
									linkToNext, &nextAccess, nextTaskInfo
								);
							}
						}
						
						return true;
					}
				);
				
				// Remove self as writer to return
				newWriters.erase(positionAndConfirmation.first);
				
			} else {
				assert((access->_type == WRITE_ACCESS_TYPE) || (access->_type == READWRITE_ACCESS_TYPE));
				
				// Either there are previous readers or writers, but not both
				assert((lastReaders.empty() != lastWriters.empty()) && newWriters.empty());
				
				// Add previous readers as predecessors
				for (task_id_t reader : lastReaders) {
					link._predecessors.insert(reader);
					
					task_info_t &readerTaskInfo = _taskToInfoMap[reader];
					if ((readerTaskInfo._parent == parentId) && (readerTaskInfo._taskGroupPhaseIndex == taskInfo._taskGroupPhaseIndex)) {
						taskInfo._hasPredecessorsInSameLevel = true;
						readerTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				// Add previous writers as predecessors
				// Either we have previous readers or writers, but not both
				for (task_id_t writer : lastWriters) {
					link._predecessors.insert(writer);
				
					task_info_t &writerTaskInfo = _taskToInfoMap[writer];
					if ((writerTaskInfo._parent == parentId) && (writerTaskInfo._taskGroupPhaseIndex == taskInfo._taskGroupPhaseIndex)) {
						taskInfo._hasPredecessorsInSameLevel = true;
						writerTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				// Current access has type writer so we flush previous writers and insert self as one
				auto positionAndConfirmation = newWriters.insert(access->_originator);
				assert(positionAndConfirmation.second);
				
				// Advance to the next access
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						DataAccessRange nextRange = range.intersect(nextAccess._accessRange);
						
						if (!nextRange.empty()) {
							predecessors_t nextLastReaders;
							predecessors_t nextNewWriters;
							predecessors_t& nextLastWriters = newWriters;
							
							buildPredecessorList(
								nextRange,
								nextLastReaders, nextLastWriters, nextNewWriters,
								linkToNext, &nextAccess, nextTaskInfo
							);
						}
						
						return true;
					}
				);
				
				// Remove self as writer to return
				newWriters.erase(positionAndConfirmation.first);
			}
		}
		
		
		static void buildPredecessorList(access_t *access)
		{
			assert(access != nullptr);
			task_info_t &originatorTaskInfo = _taskToInfoMap[access->_originator];
			task_id_t parentId = originatorTaskInfo._parent;
			
			foreachLiveNextOfAccess(access,
				[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
					DataAccessRange range = access->_accessRange.intersect(nextAccess._accessRange);
					
					if (!range.empty()) {
						predecessors_t lastReaders;
						predecessors_t lastWriters;
						predecessors_t newWriters;
						
						switch(access->_type) {
							case READ_ACCESS_TYPE:
								lastReaders.insert(access->_originator);
								break;
							case WRITE_ACCESS_TYPE:
								lastWriters.insert(access->_originator);
								break;
							case READWRITE_ACCESS_TYPE:
								lastWriters.insert(access->_originator);
								break;
							case CONCURRENT_ACCESS_TYPE:
								if (nextAccess._type == CONCURRENT_ACCESS_TYPE)
									newWriters.insert(access->_originator);
								else
									lastWriters.insert(access->_originator);
								break;
						}
						
						buildPredecessorList(
							range,
							lastReaders, lastWriters, newWriters,
							linkToNext, &nextAccess, nextTaskInfo
						);
						
						if ((nextTaskInfo._parent == parentId) &&  nextTaskInfo._hasSuccessorsInSameLevel) {
							originatorTaskInfo._hasSuccessorsInSameLevel = true;
						}
					}
					
					return true;
				}
			);
		}
		
		
		static void buildPredecessorListFromFragment(access_t *fragment)
		{
			assert(fragment != nullptr);
			assert(fragment->fragment());
			
			task_info_t &originatorTaskInfo = _taskToInfoMap[fragment->_originator];
			task_id_t parentId = originatorTaskInfo._parent;
			
			foreachLiveNextOfAccess(fragment,
				[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
					DataAccessRange range = fragment->_accessRange.intersect(nextAccess._accessRange);
					
					if (!range.empty()) {
						buildPredecessorList(&nextAccess);
					}
					
					return true;
				}
			);
		}
		
		
		static taskwait_id_t findPrecedingTaskwait(task_info_t const &taskInfo)
		{
			if (taskInfo._parent == task_id_t()) {
				return taskwait_id_t();
			}
			
			if (taskInfo._hasPredecessorsInSameLevel) {
				return taskwait_id_t();
			}
			
			task_info_t const &parent = _taskToInfoMap[taskInfo._parent];
			if (taskInfo._taskGroupPhaseIndex > 0) {
				phase_t *previousPhase = parent._phaseList[taskInfo._taskGroupPhaseIndex - 1];
				assert(previousPhase != nullptr);
				
				taskwait_t *previousTaskwait = dynamic_cast<taskwait_t *> (previousPhase);
				assert(previousTaskwait != nullptr);
				
				return previousTaskwait->_taskwaitId;
			} else {
				return findPrecedingTaskwait(parent);
			}
		}
		
		static taskwait_id_t findSuccedingTaskwait(task_info_t const &taskInfo)
		{
			if (taskInfo._parent == task_id_t()) {
				return taskwait_id_t();
			}
			
			if (taskInfo._hasSuccessorsInSameLevel) {
				return taskwait_id_t();
			}
			
			if (taskInfo._isIf0) {
				return taskwait_id_t();
			}
			
			task_info_t const &parent = _taskToInfoMap[taskInfo._parent];
			size_t nextTaskGroupPhaseIndex = taskInfo._taskGroupPhaseIndex + 1;
			if (nextTaskGroupPhaseIndex < parent._phaseList.size()) {
				phase_t *nextPhase = parent._phaseList[nextTaskGroupPhaseIndex];
				assert(nextPhase != nullptr);
				
				taskwait_t *nextTaskwait = dynamic_cast<taskwait_t *> (nextPhase);
				assert(nextTaskwait != nullptr);
				
				return nextTaskwait->_taskwaitId;
			} else {
				return findSuccedingTaskwait(parent);
			}
		}
		
		
		void generateTaskwaitRelations()
		{
			for (auto &taskIdAndTask : _taskToInfoMap) {
				task_info_t &taskInfo = taskIdAndTask.second;
				
				taskInfo._precedingTaskwait = findPrecedingTaskwait(taskInfo);
				taskInfo._succedingTaskwait = findSuccedingTaskwait(taskInfo);
			}
			
			for (auto &taskwaitIdAndTaskwait : _taskwaitToInfoMap) {
				taskwait_t *taskwait = taskwaitIdAndTaskwait.second;
				assert(taskwait != nullptr);
				
				size_t phaseIndex = taskwait->_taskPhaseIndex;
				task_info_t const &task = _taskToInfoMap[taskwait->_task];
				
				if (task._phaseList.size() > (phaseIndex + 1)) {
					taskwait_t *nextTaskwait = dynamic_cast<taskwait_t *> (task._phaseList[phaseIndex + 1]);
					if (nextTaskwait != nullptr) {
						taskwait->_immediateNextTaskwait = nextTaskwait->_taskwaitId;
					}
				}
			}
		}
		
		
		void generateEdges()
		{
			// Build the predecessor list that each access link activates
			markAccessesWithPrevious();
			for (auto idAndAccess : _accessIdToAccessMap) {
				access_t *access = idAndAccess.second;
				assert(access != nullptr);
				
// 				if (!access->fragment() && !access->hasPrevious()) {
				if (!access->fragment()) {
					buildPredecessorList(access);
				} else {
					buildPredecessorListFromFragment(access);
				}
			}
			
			// Generate the edges
			for (auto &accessIdAndPointer : _accessIdToAccessMap) {
				access_t *access = accessIdAndPointer.second;
				
				for (auto &nextTaskAndLink : access->_nextLinks) {
					task_id_t nextTaskId = nextTaskAndLink.first;
					link_to_next_t &link = nextTaskAndLink.second;
					for (task_id_t predecessorId : link._predecessors) {
						task_info_t &predecessor = _taskToInfoMap[predecessorId];
						
						// Create the edge if it does not exist yet
						predecessor._outputEdges[nextTaskId];
					}
				}
			}
			
			
// 			for (auto &idAndTask : _taskToInfoMap) {
// 				assert(idAndTask.first != task_id_t());
// 				
// 				task_info_t &taskInfo = idAndTask.second;
// 				
// 				taskInfo._liveAccesses.processAll(
// 					[&](task_live_accesses_t::iterator position) -> bool {
// 						access_t *access = position->_access;
// 						assert(access != nullptr);
// 						
// 						for (auto &nextTaskAndLink : access->_nextLinks) {
// 							task_id_t nextTaskId = nextTaskAndLink.first;
// 							link_to_next_t &link = nextTaskAndLink.second;
// 							for (task_id_t predecessorId : link._predecessors) {
// 								task_info_t &predecessor = _taskToInfoMap[predecessorId];
// 								
// 								// Create the edge if it does not exist yet
// 								predecessor._outputEdges[nextTaskId];
// 							}
// 						}
// 						
// 						return true;
// 					}
// 				);
// 			}
		}
		
	} // Instrument::Graph
} // Instrument

