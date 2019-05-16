/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "PathLength.hpp"

#include "InstrumentGraph.hpp"


namespace Instrument {
	namespace Graph {
		
		
		struct graph_node_info_t {
			int _maximumDepth;
			bool _visited;
			task_id_t _lastInLongestPath;
			
			graph_node_info_t()
			: _maximumDepth(1), _visited(false), _lastInLongestPath()
			{
			}
		};
		
		
		typedef std::map<task_id_t, graph_node_info_t> task_node_info_map_t;
		
		
		static void findDepth(task_id_t currentId, graph_node_info_t &current, task_node_info_map_t &taskNodeInfoMap)
		{
			if (current._visited) {
				return;
			}
			current._visited = true;
			
			current._maximumDepth = 1;
			current._lastInLongestPath = currentId;
			
			task_info_t &currentTaskInfo = _taskToInfoMap[currentId];
			for (auto &successorAndDependencyEdge : currentTaskInfo._outputEdges) {
				task_id_t successorId = successorAndDependencyEdge.first;
				task_info_t &successorTaskInfo = _taskToInfoMap[successorId];
				
				if (currentTaskInfo._parent != successorTaskInfo._parent) {
					// Only local paths
					continue;
				}
				
				graph_node_info_t &successor = taskNodeInfoMap[successorId];
				findDepth(successorId, successor, taskNodeInfoMap);

				
				if (successor._maximumDepth + 1 > current._maximumDepth) {
					current._maximumDepth = successor._maximumDepth + 1;
					current._lastInLongestPath = successor._lastInLongestPath;
				}
			}
		}
		
		
		void findTopmostTasksAndPathLengths(task_id_t startingTaskId)
		{
			task_info_t &startingTaskInfo = _taskToInfoMap[startingTaskId];
			
			// Skip tasks without inner subtasks
			if (startingTaskInfo._phaseList.empty()) {
				return;
			}
			
			for (unsigned int phase = 0; phase < startingTaskInfo._phaseList.size(); phase++) {
				phase_t *currentPhase = startingTaskInfo._phaseList[phase];
				
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(currentPhase);
				if (taskGroup == nullptr) {
					continue;
				}
				
				task_node_info_map_t taskNodeInfoMap;
				
				// Create one node per task
				for (auto childId : taskGroup->_children) {
					taskNodeInfoMap[childId];
				}
				
				task_id_t longestPathStart;
				task_id_t longestPathEnd;
				int longestPathLength = 0;
				for (auto &idAndNode : taskNodeInfoMap) {
					task_id_t nodeId = idAndNode.first;
					graph_node_info_t &node = idAndNode.second;
					
					findDepth(nodeId, node, taskNodeInfoMap);
					if (node._maximumDepth > longestPathLength) {
						longestPathLength = node._maximumDepth;
						longestPathStart = nodeId;
						longestPathEnd = node._lastInLongestPath;
					}
					
					findTopmostTasksAndPathLengths(nodeId);
				}
				
				taskGroup->_longestPathFirstTaskId = longestPathStart;
				taskGroup->_longestPathLastTaskId = longestPathEnd;
			}
		}
		
		
	}
}
