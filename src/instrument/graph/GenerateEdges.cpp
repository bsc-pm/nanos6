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
			task_id_t const &lastWriter, predecessors_t &lastReaders,
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
										lastWriter, lastReaders,
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
				if (lastWriter != task_id_t()) {
					link._predecessors.insert(lastWriter);
					
					task_info_t &lastWriterTaskInfo = _taskToInfoMap[lastWriter];
					if (lastWriterTaskInfo._parent == parentId) {
						taskInfo._hasPredecessorsInSameLevel = true;
						lastWriterTaskInfo._hasSuccessorsInSameLevel = true;
					}
				}
				
				auto positionAndConfirmation = lastReaders.insert(access->_originator);
				assert(positionAndConfirmation.second);
				
				// Advance to the next access
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						DataAccessRange nextRange = range.intersect(nextAccess._accessRange);
						
						if (!nextRange.empty()) {
							buildPredecessorList(
								nextRange,
								lastWriter, lastReaders,
								linkToNext, &nextAccess, nextTaskInfo
							);
						}
						
						return true;
					}
				);
				
				lastReaders.erase(positionAndConfirmation.first);
			} else {
				if (lastReaders.empty()) {
					assert(lastWriter != task_id_t());
					link._predecessors.insert(lastWriter);
					
					task_info_t &lastWriterTaskInfo = _taskToInfoMap[lastWriter];
					if (lastWriterTaskInfo._parent == parentId) {
						taskInfo._hasPredecessorsInSameLevel = true;
						lastWriterTaskInfo._hasSuccessorsInSameLevel = true;
					}
				} else {
					for (task_id_t reader : lastReaders) {
						link._predecessors.insert(reader);
						
						task_info_t &readerTaskInfo = _taskToInfoMap[reader];
						if (readerTaskInfo._parent == parentId) {
							taskInfo._hasPredecessorsInSameLevel = true;
							readerTaskInfo._hasSuccessorsInSameLevel = true;
						}
					}
				}
				
				// Advance to the next access
				predecessors_t nextReaders;
				foreachLiveNextOfAccess(access,
					[&](access_t &nextAccess, link_to_next_t &linkToNext, task_info_t &nextTaskInfo) -> bool {
						DataAccessRange nextRange = range.intersect(nextAccess._accessRange);
						
						if (!nextRange.empty()) {
							buildPredecessorList(
								nextRange,
								access->_originator, nextReaders,
								linkToNext, &nextAccess, nextTaskInfo
							);
						}
						
						return true;
					}
				);
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
						task_id_t lastWriter;
						predecessors_t lastReaders;
						
						switch(access->_type) {
							case READ_ACCESS_TYPE:
								lastReaders.insert(access->_originator);
								break;
							case WRITE_ACCESS_TYPE:
								lastWriter = access->_originator;
								break;
							case READWRITE_ACCESS_TYPE:
								lastWriter = access->_originator;
								break;
						}
						
						buildPredecessorList(
							range,
							lastWriter, lastReaders,
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

