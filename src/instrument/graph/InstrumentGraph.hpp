/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_GRAPH_HPP
#define INSTRUMENT_GRAPH_GRAPH_HPP


#include <nanos6.h>

#include "dependencies/DataAccessType.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"
#include "system/ompss/UserMutex.hpp"

#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>

#include <instrument/api/InstrumentDependenciesByAccessLinks.hpp>

#include <DataAccessRegionIndexer.hpp>

#include <atomic>
#include <bitset>
#include <cassert>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>


class WorkerThread;


namespace Instrument {
	
	namespace Graph {
		class taskwait_id_t {
		public:
			typedef long int inner_type_t;
			
		private:
			inner_type_t _id;
			
		public:
			taskwait_id_t(inner_type_t id)
			: _id(id)
			{
			}
			
			taskwait_id_t()
			: _id(-1)
			{
			}
			
			operator inner_type_t() const
			{
				return _id;
			}
		};
		
		typedef long usermutex_id_t;
		
		enum task_status_t {
			not_created_status,
			not_started_status,
			started_status,
			blocked_status,
			finished_status,
			deleted_status
		};
		
		//! \brief this is the list of direct children between the previous (if any) and next (if any) taskwait
		typedef std::set<task_id_t> children_list_t;
		
		typedef enum {
			READ = READ_ACCESS_TYPE,
			WRITE = WRITE_ACCESS_TYPE,
			READWRITE = READWRITE_ACCESS_TYPE,
			CONCURRENT = CONCURRENT_ACCESS_TYPE,
			COMMUTATIVE = COMMUTATIVE_ACCESS_TYPE,
			REDUCTION = REDUCTION_ACCESS_TYPE,
			LOCAL = NO_ACCESS_TYPE,
			NOT_CREATED
		} access_type_t;
		
		enum link_status_t {
			not_created_link_status,
			created_link_status,
			dead_link_status
		};
		
		typedef std::set<task_id_t> predecessors_t;
		
		struct link_to_next_t {
			bool _direct;
			bool _bidirectional;
			access_object_type_t _sinkObjectType;
			link_status_t _status;
			predecessors_t _predecessors; // Predecessors of next that the link "enables"
			
			link_to_next_t()
			{
				assert("Instrument::Graph did not find a link between two data accesses" == 0);
			}
			
			link_to_next_t(bool direct, bool bidirectional, access_object_type_t sinkObjectType)
				: _direct(direct), _bidirectional(bidirectional), _sinkObjectType(sinkObjectType),
				_status(not_created_link_status),
				_predecessors()
			{
			}
		};
		
		typedef std::map<task_id_t, link_to_next_t> data_access_next_links_t;
		
		enum access_status_t {
			not_created_access_status = 0,
			created_access_status,
			removable_access_status,
			removed_access_status
		};
		
		struct access_t {
			enum {
				ACCESS_WEAK_BIT=0,
				ACCESS_SATISFIED_BIT,
				ACCESS_COMPLETE_BIT,
				ACCESS_HAS_PREVIOUS_BIT,
				ACCESS_BITSET_SIZE
			};
			typedef std::bitset<ACCESS_BITSET_SIZE> bitset_t;
			
			access_object_type_t _objectType;
			data_access_id_t _id;
			data_access_id_t _superAccess;
			DataAccessType _type;
			DataAccessRegion _accessRegion;
			task_id_t _originator;
			int _parentPhase;
			
			bitset_t _bitset;
			std::set<std::string> _otherProperties;
			
			data_access_next_links_t _nextLinks;
			access_status_t _status;
			
			// A group of accesses is a set of accesses that have originated as fragments of an initial access
			data_access_id_t _firstGroupAccess; // The initial access
			data_access_id_t _nextGroupAccess; // The next fragment of the original access
			
			access_t(access_object_type_t objectType):
				_objectType(objectType),
				_id(),
				_superAccess(),
				_type(READ_ACCESS_TYPE), _accessRegion(),
				_originator(-1),
				_parentPhase(-1),
				_bitset(),
				_nextLinks(),
				_status(not_created_access_status),
				_firstGroupAccess(), _nextGroupAccess()
			{
			}
			
			DataAccessRegion const &getAccessRegion() const
			{
				return _accessRegion;
			}
			DataAccessRegion &getAccessRegion()
			{
				return _accessRegion;
			}
			
			bool weak() const
			{
				return _bitset[ACCESS_WEAK_BIT];
			}
			
			bitset_t::reference weak()
			{
				return _bitset[ACCESS_WEAK_BIT];
			}
			
			
			bool satisfied() const
			{
				return _bitset[ACCESS_SATISFIED_BIT];
			}
			
			bitset_t::reference satisfied()
			{
				return _bitset[ACCESS_SATISFIED_BIT];
			}
			
			
			bool complete() const
			{
				return _bitset[ACCESS_COMPLETE_BIT];
			}
			
			bitset_t::reference complete()
			{
				return _bitset[ACCESS_COMPLETE_BIT];
			}
			
			
			bool hasPrevious() const
			{
				return _bitset[ACCESS_HAS_PREVIOUS_BIT];
			}
			
			bitset_t::reference hasPrevious()
			{
				return _bitset[ACCESS_HAS_PREVIOUS_BIT];
			}
		};
		
		
		struct task_group_t;
		struct access_fragment_t : public access_t {
			task_group_t *_taskGroup;
			
			access_fragment_t(access_object_type_t objectType)
				: access_t(objectType), _taskGroup(nullptr)
			{
				assert(objectType == entry_fragment_type);
			}
		};
		
		struct AccessFragmentWrapper {
			access_fragment_t *_fragment;
			
			AccessFragmentWrapper(access_fragment_t *fragment)
			: _fragment(fragment)
			{
				assert(fragment != nullptr);
			}
			
			DataAccessRegion const &getAccessRegion() const
			{
				return _fragment->_accessRegion;
			}
			
			DataAccessRegion &getAccessRegion()
			{
				return _fragment->_accessRegion;
			}
		};
		
		typedef DataAccessRegionIndexer<AccessFragmentWrapper> task_live_access_fragments_t;
		typedef std::set<access_fragment_t *> task_all_access_fragments_t;
		
		
		struct taskwait_fragment_t : public access_t {
			task_group_t *_taskGroup;
			
			taskwait_fragment_t(access_object_type_t objectType)
				: access_t(objectType), _taskGroup(nullptr)
			{
				assert((objectType == taskwait_type) || (objectType == top_level_sink_type));
			}
		};
		
		struct TaskwaitFragmentWrapper {
			taskwait_fragment_t *_taskwaitFragment;
			
			TaskwaitFragmentWrapper(taskwait_fragment_t *taskwaitFragment)
			: _taskwaitFragment(taskwaitFragment)
			{
				assert(taskwaitFragment != nullptr);
			}
			
			DataAccessRegion const &getAccessRegion() const
			{
				return _taskwaitFragment->_accessRegion;
			}
			
			DataAccessRegion &getAccessRegion()
			{
				return _taskwaitFragment->_accessRegion;
			}
		};
		
		
		typedef DataAccessRegionIndexer<TaskwaitFragmentWrapper> task_live_taskwait_fragments_t;
		typedef std::set<taskwait_fragment_t *> task_all_taskwait_fragments_t;
		
		
		struct phase_t {
			taskwait_id_t _nextTaskwaitId;
			
			virtual ~phase_t()
			{
			}
			
		};
		
		struct task_group_t : public phase_t {
			children_list_t _children;
			task_id_t _longestPathFirstTaskId;
			task_id_t _longestPathLastTaskId;
			taskwait_id_t _clusterId;
			task_live_access_fragments_t _liveFragments;
			task_all_access_fragments_t _allFragments;
			task_live_taskwait_fragments_t _liveTaskwaitFragments;
			task_all_taskwait_fragments_t _allTaskwaitFragments;
			
			task_group_t(taskwait_id_t id)
				: phase_t(),
				_children(),
				_longestPathFirstTaskId(), _longestPathLastTaskId(),
				_clusterId(id),
				_liveFragments(), _allFragments(),
				_liveTaskwaitFragments(), _allTaskwaitFragments()
			{
			}
		};
		
		struct taskwait_t : public phase_t {
			taskwait_id_t _taskwaitId;
			char const *_taskwaitSource;
			
			task_id_t _task;
			size_t _taskPhaseIndex;
			
			task_status_t _status;
			long _lastCPU;
			
			taskwait_id_t _immediateNextTaskwait;
			
			task_id_t _if0Task;
			
			taskwait_t(taskwait_id_t taskwaitId, char const *taskwaitSource, task_id_t if0Task)
				: _taskwaitId(taskwaitId), _taskwaitSource(taskwaitSource),
				_task(), _taskPhaseIndex(~0UL),
				_status(not_created_status), _lastCPU(-1),
				_immediateNextTaskwait(),
				_if0Task(if0Task)
			{
			}
		};
		
		struct AccessWrapper {
			access_t *_access;
			
			AccessWrapper(access_t *access)
				: _access(access)
			{
				assert(access != nullptr);
			}
			
			DataAccessRegion const &getAccessRegion() const
			{
				return _access->_accessRegion;
			}
			
			DataAccessRegion &getAccessRegion()
			{
				return _access->_accessRegion;
			}
		};
		
		
		typedef std::vector<phase_t *> phase_list_t;
		
		typedef DataAccessRegionIndexer<AccessWrapper> task_live_accesses_t;
		typedef std::set<access_t *> task_all_accesses_t;
		
		struct dependency_edge_t {
			bool _hasBeenMaterialized;
			int _activeStrongContributorLinks;
			int _activeWeakContributorLinks;
			
			dependency_edge_t()
				: _hasBeenMaterialized(false), _activeStrongContributorLinks(0), _activeWeakContributorLinks(0)
			{
			}
		};
		typedef std::map<task_id_t, dependency_edge_t> output_edges_t;
		
		struct task_info_t {
			nanos6_task_info_t *_nanos6_task_info;
			nanos6_task_invocation_info_t *_nanos6_task_invocation_info;
			
			task_id_t _parent;
			size_t _taskGroupPhaseIndex;
			
			task_status_t _status;
			long _lastCPU;
			
			phase_list_t _phaseList;
			
			task_live_accesses_t _liveAccesses;
			task_all_accesses_t _allAccesses;
			
			output_edges_t _outputEdges;
			
			bool _hasChildren;
			
			bool _hasPredecessorsInSameLevel;
			bool _hasSuccessorsInSameLevel;
			
			bool _isIf0;
			
			taskwait_id_t _precedingTaskwait;
			taskwait_id_t _succedingTaskwait;
			
			task_info_t()
				: _nanos6_task_info(nullptr), _nanos6_task_invocation_info(nullptr),
				_parent(-1), _taskGroupPhaseIndex(0),
				_status(not_created_status), _lastCPU(-1),
				_phaseList(),
				_liveAccesses(), _allAccesses(),
				_outputEdges(),
				_hasChildren(false),
				_hasPredecessorsInSameLevel(false), _hasSuccessorsInSameLevel(false),
				_isIf0(false),
				_precedingTaskwait(), _succedingTaskwait()
			{
			}
		};
		
		//! \brief maps tasks to their information
		typedef std::map<task_id_t, task_info_t> task_to_info_map_t;
		
		//! \brief maps task invocations to the text to use as label
		typedef std::map<nanos6_task_invocation_info_t *, std::string> task_invocation_info_label_map_t;
		
		
		struct execution_step_t;
		
		typedef std::list<execution_step_t *> execution_sequence_t;
		
		typedef std::map<data_access_id_t, access_t *> data_access_map_t;
		
		
		struct taskwait_status_t {
			task_status_t _status;
			long _lastCPU;
			
			taskwait_status_t()
			: _status(not_created_status), _lastCPU(-1)
			{
			}
		};
		
		extern std::map<taskwait_id_t, taskwait_t *> _taskwaitToInfoMap;
		
		extern std::atomic<taskwait_id_t::inner_type_t> _nextTaskwaitId;
		extern std::atomic<task_id_t::inner_type_t> _nextTaskId;
		extern std::atomic<usermutex_id_t> _nextUsermutexId;
		extern std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId;
		
		
		//! \brief maps task identifiers to their information
		extern task_to_info_map_t _taskToInfoMap;
		
		//! \brief maps data access identifiers to their associated access
		extern data_access_map_t _accessIdToAccessMap;
		
		//! \brief maps task invocation struct addresses to the text to use as task label
		extern task_invocation_info_label_map_t _taskInvocationLabel;
		
		typedef std::map<UserMutex *, usermutex_id_t> usermutex_to_id_map_t;
		
		//! \brief maps user mutexes to a numeric identifier
		extern usermutex_to_id_map_t _usermutexToId;
		
		//! \brief sequence of task executions with their corresponding CPU
		extern execution_sequence_t _executionSequence;
		
		extern SpinLock _graphLock;
		
		extern EnvironmentVariable<bool> _showDependencyStructures;
		extern EnvironmentVariable<bool> _showRegions;
		extern EnvironmentVariable<bool> _showLog;
		
		
		
		// Helper functions
		template <typename ProcessorType>
		static inline void foreachLiveNextOfAccess(access_t *access, ProcessorType processor)
		{
			assert(access != nullptr);
			
			for (auto &nextAndLink : access->_nextLinks) {
				task_id_t nextTaskId = nextAndLink.first;
				link_to_next_t &linkToNext = nextAndLink.second;
				
				if ((linkToNext._sinkObjectType == regular_access_type) || (linkToNext._sinkObjectType == entry_fragment_type)) {
					task_info_t &nextTaskInfo = _taskToInfoMap[nextTaskId];
					
					nextTaskInfo._liveAccesses.processIntersecting(
						access->_accessRegion,
						[&](task_live_accesses_t::iterator position) -> bool {
							access_t *nextAccess = position->_access;
							assert(nextAccess != nullptr);
							
							return processor(*nextAccess, linkToNext, nextTaskInfo);
						}
					);
					
					if (!nextTaskInfo._phaseList.empty()) {
						// Forward through the fragments of the first task group (that is, limited up to the first barrier)
						task_group_t *taskGroup = dynamic_cast<task_group_t *> (*nextTaskInfo._phaseList.begin());
						if (taskGroup != nullptr) {
							taskGroup->_liveFragments.processIntersecting(
								access->_accessRegion,
								[&](task_live_access_fragments_t::iterator fragmentPosition) -> bool {
									access_fragment_t *fragment = fragmentPosition->_fragment;
									assert(fragment != nullptr);
									assert(fragment->_objectType == entry_fragment_type);
									
									foreachLiveNextOfAccess(fragment, processor);
									
									return true;
								}
							);
						}
					}
				} else {
					assert((linkToNext._sinkObjectType == taskwait_type) || (linkToNext._sinkObjectType == top_level_sink_type));
					
					task_info_t *ancestorTaskInfo = &_taskToInfoMap[access->_originator]; // Originator TaskInfo
					while (ancestorTaskInfo->_parent != nextTaskId) {
						ancestorTaskInfo = &_taskToInfoMap[ancestorTaskInfo->_parent];
					}
					
					task_info_t &nextTaskInfo = _taskToInfoMap[nextTaskId];
					// Next's taskGroup phase index: In a taskwait/top-level sink it corresponds to
					// a taskgroup immediately previous to the taskwait, which can be found in the
					// ancestor path starting from the current 'access'
					size_t nextTaskGroupPhaseIndex = ancestorTaskInfo->_taskGroupPhaseIndex;
					
					task_group_t *nextTaskGroup = dynamic_cast<task_group_t *> (nextTaskInfo._phaseList[nextTaskGroupPhaseIndex]);
					assert(nextTaskGroup != nullptr);
					
					nextTaskGroup->_liveTaskwaitFragments.processIntersecting(
						access->_accessRegion,
						[&](task_live_taskwait_fragments_t::iterator taskwaitFragmentPosition) -> bool {
							taskwait_fragment_t *taskwaitFragment = taskwaitFragmentPosition->_taskwaitFragment;
							assert(taskwaitFragment != nullptr);
							
							return processor(*taskwaitFragment, linkToNext, nextTaskInfo);
						}
					);
				}
			}
		}
		
		
		template <typename ProcessorType>
		static inline void foreachItersectingNextOfAccess(access_t *access, ProcessorType processor)
		{
			assert(access != nullptr);
			
			for (auto &nextAndLink : access->_nextLinks) {
				task_id_t nextTaskId = nextAndLink.first;
				link_to_next_t &linkToNext = nextAndLink.second;
				
				if ((linkToNext._sinkObjectType == regular_access_type) || (linkToNext._sinkObjectType == entry_fragment_type)) {
					task_info_t &nextTaskInfo = _taskToInfoMap[nextTaskId];
					for (access_t *nextAccess : nextTaskInfo._allAccesses) {
						assert (nextAccess != nullptr);
						
						if (access->_accessRegion.intersect(nextAccess->_accessRegion).empty()) {
							continue;
						}
						
						bool result = processor(*nextAccess, linkToNext, nextTaskInfo);
						if (!result) {
							break;
						}
					}
				} else {
					assert((linkToNext._sinkObjectType == taskwait_type) || (linkToNext._sinkObjectType == top_level_sink_type));
					
					task_info_t *ancestorTaskInfo = &_taskToInfoMap[access->_originator]; // Originator TaskInfo
					while (ancestorTaskInfo->_parent != nextTaskId) {
						ancestorTaskInfo = &_taskToInfoMap[ancestorTaskInfo->_parent];
					}
					
					task_info_t &nextTaskInfo = _taskToInfoMap[nextTaskId];
					// Next's taskGroup phase index: In a taskwait/top-level sink it corresponds to
					// a taskgroup immediately previous to the taskwait, which can be found in the
					// ancestor path starting from the current 'access'
					size_t nextTaskGroupPhaseIndex = ancestorTaskInfo->_taskGroupPhaseIndex;
					
					task_group_t *nextTaskGroup = dynamic_cast<task_group_t *> (nextTaskInfo._phaseList[nextTaskGroupPhaseIndex]);
					assert(nextTaskGroup != nullptr);
					
					for (taskwait_fragment_t *taskwaitFragment : nextTaskGroup->_allTaskwaitFragments) {
						assert(taskwaitFragment != nullptr);
						
						if (access->_accessRegion.intersect(taskwaitFragment->_accessRegion).empty()) {
							continue;
						}
						
						bool result = processor(*taskwaitFragment, linkToNext, nextTaskInfo);
						if (!result) {
							break;
						}
					}
				}
			}
		}
		
		
	};
	
}


#endif // INSTRUMENT_GRAPH_GRAPH_HPP
