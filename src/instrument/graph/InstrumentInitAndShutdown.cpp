#include "InstrumentInitAndShutdown.hpp"

#include "InstrumentGraph.hpp"
#include "Color.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

#include <fstream>
#include <sstream>

#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>


namespace Instrument {
	using namespace Graph;
	
	
	void initialize()
	{
		// Assign thread identifier 0 to the leader thread
		_threadToId[0] = 0;
	}
	
	
	static long _nextCluster = 1;
	
	static EnvironmentVariable<bool> _shortenFilenames("NANOS_GRAPH_SHORTEN_FILENAMES", false);
	static EnvironmentVariable<bool> _showSpuriousDependencyStructures("NANOS_GRAPH_SHOW_SPURIOUS_DEPENDENCY_STRUCTURES", false);
	static EnvironmentVariable<bool> _showDeadDependencyStructures("NANOS_GRAPH_SHOW_DEAD_DEPENDENCY_STRUCTURES", false);
	static EnvironmentVariable<bool> _showAllSteps("NANOS_GRAPH_SHOW_ALL_STEPS", false);
	
	
	static inline bool isComposite(task_id_t taskId)
	{
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		return (!taskInfo._phaseList.empty());
	}
	
	static inline void shortenString(std::string /* INOUT */ &s)
	{
		size_t lastSlash = s.rfind('/');
		if (lastSlash != std::string::npos) {
			s = s.substr(lastSlash+1);
		}
	}
	
	static inline std::string const &getTaskName(task_info_t &taskInfo)
	{
		assert(taskInfo._nanos_task_invocation_info != nullptr);
		task_invocation_info_label_map_t::iterator it = _taskInvocationLabel.find(taskInfo._nanos_task_invocation_info);
		if (it != _taskInvocationLabel.end()) {
			return it->second;
		}
		
		std::string label;
		if (taskInfo._nanos_task_info->task_label != nullptr) {
			label = taskInfo._nanos_task_info->task_label;
		} else if ((taskInfo._nanos_task_invocation_info != nullptr) && (taskInfo._nanos_task_invocation_info->invocation_source != nullptr)) {
			label = taskInfo._nanos_task_invocation_info->invocation_source;
		} else if (taskInfo._nanos_task_info->declaration_source != nullptr) {
			label = taskInfo._nanos_task_info->declaration_source;
		} else {
			label = std::string();
		}
		
		if (_shortenFilenames) {
			shortenString(label);
		}
		
		_taskInvocationLabel[taskInfo._nanos_task_invocation_info] = std::move(label);
		return _taskInvocationLabel[taskInfo._nanos_task_invocation_info];
	}
	
	static std::string indentation;
	
	
	static std::string makeTaskLabel(task_id_t id, task_info_t &taskInfo)
	{
		std::ostringstream oss;
		
		oss << id;
		std::string taskName = getTaskName(taskInfo);
		if (!taskName.empty()) {
			oss << "\\n" << taskName;
		}
		
		return oss.str();
	}
	
	
	static inline std::string makeTaskNodeName(task_id_t taskId)
	{
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		std::ostringstream oss;
		
		if (taskInfo._phaseList.empty()) {
			oss << "task";
		} else {
			oss << "cluster_task";
		}
		oss << taskId;
		
		return oss.str();
	}
	
	
	static std::string makeTaskwaitLabel(char const *sourceLocation)
	{
		std::ostringstream oss;
		
		oss << "TASKWAIT";
		if ((sourceLocation != nullptr) && (std::string() != sourceLocation)) {
			std::string location(sourceLocation);
			if (_shortenFilenames) {
				shortenString(location);
			}
			
			oss << "\\n" << location;
		}
		
		return oss.str();
	}
	
	
	struct taskwait_status_t {
		task_status_t _status;
		long _lastCPU;
		
		taskwait_status_t()
			: _status(not_created_status), _lastCPU(-1)
		{
		}
	};
	
	static std::map<taskwait_id_t, taskwait_status_t> _taskwaitStatus;
	
	
	static std::string getTaskAttributes(task_status_t status, long cpu)
	{
		std::ostringstream oss;
		
		switch (status) {
			case not_created_status:
				oss << " style=\"filled\" penwidth=1 color=\"#888888\" fillcolor=\"white\" ";
				break;
			case not_started_status:
				oss << " style=\"filled\" penwidth=2 fillcolor=\"white\" ";
				break;
			case started_status:
				assert(cpu != -1);
				oss << " style=\"filled\" penwidth=1 fillcolor=\"" << getColor(cpu) << "\" ";
				break;
			case blocked_status:
				assert(cpu != -1);
				oss << " style=\"filled\" penwidth=1 fillcolor=\"" << getBrightColor(cpu, 0.75) << "\" ";
				break;
			case finished_status:
				assert(cpu != -1);
				oss << " style=\"filled\" penwidth=1 color=\"#888888\" fillcolor=\"" << getBrightColor(cpu, 0.75) << "\" ";
				break;
			default:
				assert(false);
		}
		
		return oss.str();
	}
	
	
	static std::string getEdgeAttributes(task_status_t sourceStatus, task_status_t sinkStatus)
	{
		bool grayout = false;
		
		grayout |= (sourceStatus == not_created_status);
		grayout |= (sinkStatus == not_created_status);
		grayout |= (sourceStatus == finished_status);
		grayout |= (sinkStatus == finished_status);
		
		if (grayout) {
			return " color=\"#888888\" fillcolor=\"#888888\" ";
		} else {
			return "";
		}
	}
	
	
	typedef char Bool;
	
	
	static void emitTask(
		std::ofstream &ofs, task_id_t taskId,
		std::string /* OUT */ &taskLink,
		std::string /* OUT */ &sourceLink, std::string /* OUT */ &sinkLink,
		Bool /* OUT */ &hasPredecessors, Bool /* OUT */ &hasSuccessors)
	{
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		hasPredecessors = taskInfo._hasPredecessors;
		hasSuccessors = taskInfo._hasSuccessors;
		
		if (!isComposite(taskId)) {
			// A leaf task
			std::ostringstream oss;
			oss << "task" << taskId;
			taskLink = oss.str();
			
			ofs
				<< indentation << taskLink
				<< " [ label=\"" << makeTaskLabel(taskId, taskInfo) << "\" "
				<< " shape=\"box\" "
				<< getTaskAttributes(taskInfo._status, taskInfo._lastCPU)
				<< " ]" << std::endl;
			
			sourceLink = taskLink;
			sinkLink = taskLink;
		} else {
			bool sourceLinkIsSet = false;
			
			std::ostringstream oss;
			oss << "cluster_task" << taskId;
			taskLink = oss.str();
			
			std::string initialIndentation = indentation;
			
			ofs << indentation << "subgraph " << taskLink << " {" << std::endl;
			indentation = initialIndentation + "\t";
			ofs << indentation << "label=\"" << makeTaskLabel(taskId, taskInfo) << "\";" << std::endl;
			ofs << indentation << "compound=true;" << std::endl;
			ofs << indentation << "color=\"black\";" << std::endl;
			ofs << indentation << "penwidth=1 ;" << std::endl;
			ofs << indentation << getTaskAttributes(taskInfo._status, taskInfo._lastCPU) << std::endl;
			
			std::vector<std::string> previousPhaseLinks;
			std::vector<std::string> previousPhaseSinkLinks;
			std::vector<Bool> previousPhaseElementsHaveSuccessors;
			std::vector<task_status_t> previousPhaseStatuses;
			for (unsigned int phase = 0; phase < taskInfo._phaseList.size(); phase++) {
				phase_t *currentPhase = taskInfo._phaseList[phase];
				
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(currentPhase);
				taskwait_t *taskWait = dynamic_cast<taskwait_t *>(currentPhase);
				
				size_t phaseElements = 1;
				if (taskGroup != nullptr) {
					phaseElements = taskGroup->_children.size();
				}
				
				std::vector<std::string> currentPhaseLinks(phaseElements);
				std::vector<std::string> currentPhaseSourceLinks(phaseElements);
				std::vector<std::string> currentPhaseSinkLinks(phaseElements);
				std::vector<Bool> currentPhaseElementsHavePredecessors(phaseElements);
				std::vector<Bool> currentPhaseElementsHaveSuccessors(phaseElements);
				std::vector<task_status_t> currentPhaseStatuses(phaseElements);
				if (taskGroup != nullptr) {
					long currentCluster = _nextCluster++;
					
					std::ostringstream oss;
					oss << "cluster_phase" << currentCluster;
					std::string currentPhaseLink = oss.str();
					
					ofs << indentation << "subgraph " << currentPhaseLink << " {" << std::endl;
					indentation = initialIndentation + "\t\t";
					ofs << indentation << "label=\"\";" << std::endl;
					ofs << indentation << "rank=same;" << std::endl;
					ofs << indentation << "compound=true;" << std::endl;
					ofs << indentation << "style=\"invisible\";" << std::endl;
					
					for (data_access_id_t sourceAccessId : taskGroup->_dataAccesses) {
						access_t &sourceAccess = _accessIdToAccessMap[sourceAccessId];
						
						if (!_showDeadDependencyStructures && (sourceAccess._deleted || (sourceAccess._type == NOT_CREATED))) {
							// Skip dead accesses
							continue;
						}
						
						ofs << indentation << "data_access_" << sourceAccessId << " [ shape=ellipse";
						switch (sourceAccess._type) {
							case READ:
								ofs << " label=\"R\" style=\"filled,dashed\"";
								if (sourceAccess._deleted) {
									ofs << " fillcolor=\"#AAFFAA\"";
								} else {
									ofs << " fillcolor=\"#00FF00\"";
								}
								break;
							case WRITE:
								ofs << " label=\"W\" style=\"filled,dashed\"";
								if (sourceAccess._deleted) {
									ofs << " fillcolor=\"#FFAAAA\"";
								} else {
									ofs << " fillcolor=\"#FF0000\"";
								}
								break;
							case READWRITE:
								ofs << " label=\"RW\" style=\"filled,dashed\"";
								if (sourceAccess._deleted) {
									ofs << " fillcolor=\"#AAFFAA;0.5:#FFAAAA\"";
								} else {
									ofs << " fillcolor=\"#00FF00;0.5:#FF0000\"";
								}
								break;
							case NOT_CREATED:
								ofs << " label=\"--\" style=\"filled,dashed\" fillcolor=\"#AAAAAA\"";
								assert(!sourceAccess._deleted);
								break;
						}
						
						if (sourceAccess._satisfied) {
							ofs << " penwidth=2";
						} else {
							ofs << " penwidth=1";
						}
						ofs << " ]" << std::endl;
					}
					
					std::map<task_id_t, size_t> taskId2Index;
					{
						size_t index = 0;
						for (task_id_t childId : taskGroup->_children) {
							emitTask(
								ofs, childId,
								currentPhaseLinks[index],
								currentPhaseSourceLinks[index], currentPhaseSinkLinks[index],
								currentPhaseElementsHavePredecessors[index], currentPhaseElementsHaveSuccessors[index]
							);
							
							taskId2Index[childId] = index;
							task_info_t const &childInfo = _taskToInfoMap[childId];
							currentPhaseStatuses[index] = childInfo._status;
							
							index++;
						}
					}
					
					for (edge_t edge : taskGroup->_dependencyEdges) {
						size_t sourceIndex = taskId2Index[edge._source];
						size_t sinkIndex = taskId2Index[edge._sink];
						
						ofs << indentation << currentPhaseSinkLinks[sourceIndex] << " -> " << currentPhaseSourceLinks[sinkIndex];
						ofs << " [";
						if (currentPhaseSinkLinks[sourceIndex] != currentPhaseLinks[sourceIndex]) {
							ofs << " ltail=\"" << currentPhaseLinks[sourceIndex] << "\"";
						}
						if (currentPhaseSourceLinks[sinkIndex] != currentPhaseLinks[sinkIndex]) {
							ofs << " lhead=\"" << currentPhaseLinks[sinkIndex] << "\"";
						}
						
						task_info_t const &sourceInfo = _taskToInfoMap[edge._source];
						task_info_t const &sinkInfo = _taskToInfoMap[edge._sink];
						
						if ((sourceInfo._status == not_created_status) || (sourceInfo._status == finished_status)
							|| (sinkInfo._status == not_created_status) || (sinkInfo._status == finished_status))
						{
							ofs << " color=\"#888888\" fillcolor=\"#888888\"" << std::endl;
						}
						
						ofs << " weight=1 ]" << std::endl;
					}
					
					for (data_access_id_t sourceAccessId : taskGroup->_dataAccesses) {
						access_t &sourceAccess = _accessIdToAccessMap[sourceAccessId];
						
						if (!_showDeadDependencyStructures && (sourceAccess._deleted || (sourceAccess._type == NOT_CREATED))) {
							// Skip dead accesses
							continue;
						}
						
						if (!sourceLinkIsSet && sourceAccess._previousLinks.empty()) {
							task_info_t const &originator = _taskToInfoMap[sourceAccess._originator];
							
							if (!originator._hasPredecessors) {
								std::ostringstream oss;
								oss << "data_access_" << sourceAccessId;
								sourceLink = oss.str();
								sourceLinkIsSet = true;
							}
						}
						
						{
							auto dataAccessSubLink = _firstSubAccessByAccess.find(sourceAccessId);
							if (dataAccessSubLink != _firstSubAccessByAccess.end()) {
								ofs << indentation << "data_access_" << dataAccessSubLink->first << " -> " << "data_access_" << dataAccessSubLink->second << " [ style=dotted color=\"#888888\" fillcolor=\"#888888\" weight=4 ]" << std::endl;
							}
						}
						
						assert(sourceAccess._originator != -1);
						size_t originatorIndex = taskId2Index[sourceAccess._originator];
						ofs << indentation << "data_access_" << sourceAccessId << " -> " << currentPhaseSourceLinks[originatorIndex];
						ofs << " [";
						if (currentPhaseLinks[originatorIndex] != currentPhaseSourceLinks[originatorIndex]) {
							ofs << " lhead=\"" << currentPhaseLinks[originatorIndex] << "\"";
						}
						if ((sourceAccess._type == NOT_CREATED) || sourceAccess._deleted) {
							ofs << " style=\"invis\"";
						} else if (sourceAccess._satisfied) {
							ofs << " style=dashed color=\"#888888\" fillcolor=\"#888888\"";
						} else {
							ofs << " style=dashed color=\"#000000\" fillcolor=\"#000000\"";
						}
						ofs << " weight=1 ]" << std::endl;
						
						// Links from this access
						for (auto nextLink : sourceAccess._nextLinks) {
							data_access_id_t sinkAccessId = nextLink.first;
							bool direct = nextLink.second._direct;
							
							access_t const &sinkAccess = getAccess(sinkAccessId);
							
							if (!_showDeadDependencyStructures && (sinkAccess._deleted || (sinkAccess._type == NOT_CREATED))) {
								continue;
							}
							
							if (sinkAccess._taskNestingLevel < sourceAccess._taskNestingLevel) {
								continue;
							}
							
							ofs << indentation << "data_access_" << sourceAccessId << " -> data_access_" << sinkAccessId << " [ weight=8";
							if ((nextLink.second._status == link_to_next_t::not_created_link_status) || (nextLink.second._status == link_to_next_t::dead_link_status)) {
								ofs << " style=\"invis\"";
							} else if (!direct) {
								ofs << " style=dashed color=\"#888888\" fillcolor=\"#888888\"";
							} else {
								ofs << " style=dashed color=\"#000000\" fillcolor=\"#000000\"";
							}
							ofs << " ]" << std::endl;
						}
						
						// Links to this access
						for (data_access_id_t previousAccessId: sourceAccess._previousLinks) {
							access_t &previousAccess = getAccess(previousAccessId);
							
							if (!_showDeadDependencyStructures && (previousAccess._deleted || (previousAccess._type == NOT_CREATED))) {
								continue;
							}
							
							if (sourceAccess._taskNestingLevel <= previousAccess._taskNestingLevel) {
								continue;
							}
							
							auto nextLinkPosition = previousAccess._nextLinks.find(sourceAccessId);
							assert(nextLinkPosition != previousAccess._nextLinks.end());
							auto nextLink = *nextLinkPosition;
							bool direct = nextLink.second._direct;
							
							ofs << indentation << "data_access_" << previousAccessId << " -> data_access_" << sourceAccessId << " [ weight=8";
							if ((nextLink.second._status == link_to_next_t::not_created_link_status) || (nextLink.second._status == link_to_next_t::dead_link_status)) {
								ofs << " style=\"invis\"";
							} else if (!direct) {
								ofs << " style=dashed color=\"#888888\" fillcolor=\"#888888\"";
							} else {
								ofs << " style=dashed color=\"#000000\" fillcolor=\"#000000\"";
							}
							ofs << " ]" << std::endl;
						}
					}
					
					indentation = initialIndentation + "\t";
					ofs << indentation << "}" << std::endl;
				} else if (taskWait != nullptr) {
					std::ostringstream oss;
					oss << "taskwait" << taskWait->_taskwaitId;
					std::string currentPhaseLink = oss.str();
					
					taskwait_status_t const &taskwaitStatus = _taskwaitStatus[taskWait->_taskwaitId];
					ofs
						<< indentation << currentPhaseLink
						<< " [ label=\"" << makeTaskwaitLabel(taskWait->_taskwaitSource) << "\" "
						<< getTaskAttributes(taskwaitStatus._status, taskwaitStatus._lastCPU)
						<< " shape=\"doubleoctagon\" "
						<< " ]" << std::endl;
					
					currentPhaseLinks[0] = currentPhaseLink;
					currentPhaseSourceLinks[0] = currentPhaseLink;
					currentPhaseSinkLinks[0] = currentPhaseLink;
					currentPhaseStatuses[0] = taskwaitStatus._status;
					currentPhaseElementsHavePredecessors[0] = false;
					currentPhaseElementsHaveSuccessors[0] = false;
				} else {
					assert(false);
				}
				
				if ((phase == 0) && !sourceLinkIsSet) {
					std::vector<std::string> validSourceLinks;
					for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
						if (!currentPhaseElementsHavePredecessors[currentIndex]) {
							validSourceLinks.push_back(currentPhaseSourceLinks[currentIndex]);
						}
					}
					
					sourceLink = validSourceLinks[validSourceLinks.size()/2];
					sourceLinkIsSet = true;
				}
				
				if (phase == taskInfo._phaseList.size()-1) {
					std::vector<std::string> validSinkLinks;
					for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
						if (!currentPhaseElementsHavePredecessors[currentIndex]) {
							validSinkLinks.push_back(currentPhaseSinkLinks[currentIndex]);
						}
					}
					
					sinkLink = validSinkLinks[validSinkLinks.size()/2];
				}
				
				if (!previousPhaseLinks.empty()) {
					size_t previousPhaseElements = previousPhaseLinks.size();
					
					for (size_t previousIndex=0; previousIndex < previousPhaseElements; previousIndex++) {
						if (previousPhaseElementsHaveSuccessors[previousIndex]) {
							// Link only leaf nodes of the previous phase
							continue;
						}
						
						for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
							if (currentPhaseElementsHavePredecessors[currentIndex]) {
								// Link only top nodes of the current phase
								continue;
							}
							
							ofs
								<< indentation << previousPhaseSinkLinks[previousIndex] << " -> " << currentPhaseSourceLinks[currentIndex]
								<< " [ "
								<< getEdgeAttributes(previousPhaseStatuses[previousIndex], currentPhaseStatuses[currentIndex]);
							
							if (previousPhaseLinks[previousIndex] != previousPhaseSinkLinks[previousIndex]) {
								ofs << " ltail=\"" << previousPhaseLinks[previousIndex] << "\" ";
							}
							if (currentPhaseLinks[currentIndex] != currentPhaseSourceLinks[currentIndex]) {
								ofs << " lhead=\"" << currentPhaseLinks[currentIndex] << "\" ";
							}
							ofs
								<< " weight=1 ];" << std::endl;
						}
					}
				}
				
				previousPhaseLinks = std::move(currentPhaseLinks);
				previousPhaseSinkLinks = std::move(currentPhaseSinkLinks);
				previousPhaseElementsHaveSuccessors = std::move(currentPhaseElementsHaveSuccessors);
				previousPhaseStatuses = std::move(currentPhaseStatuses);
			}
			
			indentation = initialIndentation;
			ofs << indentation << "}" << std::endl;
		}
	}
	
	
	static void dumpGraph(std::ofstream &ofs)
	{
		indentation = "";
		ofs << "digraph {" << std::endl;
		indentation = "\t";
		ofs << indentation << "compound=true;" << std::endl;
		ofs << indentation << "nanos_start [shape=Mdiamond];" << std::endl;
		ofs << indentation << "nanos_end [shape=Msquare];" << std::endl;
		
		std::string mainTaskSourceLink;
		std::string mainTaskSinkLink;
		std::string mainTaskLink;
		Bool mainTaskHasPredecessors;
		Bool mainTaskHasSuccessors;
		emitTask(ofs, 0, mainTaskLink, mainTaskSourceLink, mainTaskSinkLink, mainTaskHasPredecessors, mainTaskHasSuccessors);
		
		ofs << indentation << "nanos_start -> " << mainTaskSourceLink;
		if (mainTaskSourceLink != mainTaskLink) {
			ofs << "[ lhead=\"" << mainTaskLink << "\" ]";
		}
		ofs << ";" << std::endl;
		
		ofs << indentation << mainTaskSinkLink << " -> nanos_end";
		if (mainTaskSinkLink != mainTaskLink) {
			ofs << " [ ltail=\"" << mainTaskLink << "\" ]";
		}
		ofs << ";" << std::endl;
		ofs << "}" << std::endl;
		indentation = "";
	}
	
	
	static void emitFrame(std::string const &dir, std::string const &filenameBase, int &frame)
	{
		// Emit a graph frame
		std::ostringstream oss;
		oss << dir << "/" << filenameBase << "-step";
		oss.width(8); oss.fill('0'); oss << frame;
		oss.width(0); oss.fill(' '); oss << ".dot";
		
		std::ofstream ofs(oss.str());
		dumpGraph(ofs);
		ofs.close();
		
		frame++;
	}
	
	
	void shutdown()
	{
		std::string filenameBase;
		{
			struct timeval tv;
			gettimeofday(&tv, nullptr);
			
			std::ostringstream oss;
			oss << "graph-" << gethostid() << "-" << getpid() << "-" << tv.tv_sec;
			filenameBase = oss.str();
		}
		
		std::string dir = filenameBase + std::string("-components");
		int rc = mkdir(dir.c_str(), 0755);
		if (rc != 0) {
			FatalErrorHandler::handle(errno, " trying to create directory '", dir, "'");
		}
		
		int frame=0;
		emitFrame(dir, filenameBase, frame);
		
		task_id_t accumulateStepsTriggeredByTask = -1;
		for (execution_step_t *executionStep : _executionSequence) {
			assert(executionStep != nullptr);
			
			if (_showAllSteps) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
			}
			
			// Get the kind of step
			create_task_step_t *createTask = dynamic_cast<create_task_step_t *> (executionStep);
			enter_task_step_t *enterTask = dynamic_cast<enter_task_step_t *> (executionStep);
			exit_task_step_t *exitTask = dynamic_cast<exit_task_step_t *> (executionStep);
			
			enter_taskwait_step_t *enterTaskwait = dynamic_cast<enter_taskwait_step_t *> (executionStep);
			exit_taskwait_step_t *exitTaskwait = dynamic_cast<exit_taskwait_step_t *> (executionStep);
			
			enter_usermutex_step_t *enterUsermutex = dynamic_cast<enter_usermutex_step_t *> (executionStep);
			block_on_usermutex_step_t *blockedOnUsermutex = dynamic_cast<block_on_usermutex_step_t *> (executionStep);
			exit_usermutex_step_t *exitUsermutex = dynamic_cast<exit_usermutex_step_t *> (executionStep);
			
			create_data_access_step_t *createDataAccess = dynamic_cast<create_data_access_step_t *> (executionStep);
			upgrade_data_access_step_t *upgradeDataAccess = dynamic_cast<upgrade_data_access_step_t *> (executionStep);
			data_access_becomes_satisfied_step_t *dataAccessBecomesSatisfied = dynamic_cast<data_access_becomes_satisfied_step_t *> (executionStep);
			removed_data_access_step_t *removedDataAccess = dynamic_cast<removed_data_access_step_t *> (executionStep);
			linked_data_accesses_step_t *linkedDataAccess = dynamic_cast<linked_data_accesses_step_t *> (executionStep);
			unlinked_data_accesses_step_t *unlinkedDataAccess = dynamic_cast<unlinked_data_accesses_step_t *> (executionStep);
			
			// Update the status
			if (createTask != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = createTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == not_created_status);
				taskInfo._status = not_started_status;
				taskInfo._lastCPU = createTask->_cpu;
				
				accumulateStepsTriggeredByTask = taskId;
			} else if (enterTask != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = enterTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert((taskInfo._status == not_started_status) || (taskInfo._status == blocked_status));
				taskInfo._status = started_status;
				taskInfo._lastCPU = enterTask->_cpu;
			} else if (exitTask != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = exitTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				taskInfo._status = finished_status;
				taskInfo._lastCPU = exitTask->_cpu;
				
				accumulateStepsTriggeredByTask = taskId;
			} else if (enterTaskwait != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				taskwait_id_t taskwaitId = enterTaskwait->_taskwaitId;
				taskwait_status_t &taskwaitStatus = _taskwaitStatus[taskwaitId];
				taskwaitStatus._status = started_status;
				taskwaitStatus._lastCPU = enterTaskwait->_cpu;
				
				task_id_t taskId = enterTaskwait->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				taskInfo._status = blocked_status;
				taskInfo._lastCPU = enterTaskwait->_cpu;
			} else if (exitTaskwait != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				taskwait_id_t taskwaitId = exitTaskwait->_taskwaitId;
				taskwait_status_t &taskwaitStatus = _taskwaitStatus[taskwaitId];
				taskwaitStatus._status = finished_status;
				taskwaitStatus._lastCPU = exitTaskwait->_cpu;
				
				task_id_t taskId = exitTaskwait->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == blocked_status);
				taskInfo._status = started_status;
				taskInfo._lastCPU = exitTaskwait->_cpu;
			} else if (enterUsermutex != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = enterUsermutex->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert((taskInfo._status == started_status) || (taskInfo._status == blocked_status));
				taskInfo._status = started_status;
				taskInfo._lastCPU = enterUsermutex->_cpu;
			} else if (blockedOnUsermutex != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = blockedOnUsermutex->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				taskInfo._status = started_status;
				taskInfo._lastCPU = blockedOnUsermutex->_cpu;
			} else if (exitUsermutex != nullptr) {
				if (accumulateStepsTriggeredByTask != -1) {
					emitFrame(dir, filenameBase, frame);
					accumulateStepsTriggeredByTask = -1;
				}
				
				task_id_t taskId = exitUsermutex->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				if (taskInfo._lastCPU == exitUsermutex->_cpu) {
					// Not doing anything for now
					// Perhaps will represent the state of the mutex, its allocation slots,
					// and links from those to task-internal critical nodes
					continue;
				}
				taskInfo._lastCPU = exitUsermutex->_cpu;
			} else if (createDataAccess != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != createDataAccess->_originatorTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(createDataAccess->_accessId);
				
				access._originator = createDataAccess->_originatorTaskId;
				access._type = (access_type_t) createDataAccess->_accessType;
				access._satisfied = createDataAccess->_satisfied;
				
				accumulateStepsTriggeredByTask = createDataAccess->_originatorTaskId;
			} else if (upgradeDataAccess != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != upgradeDataAccess->_originatorTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(upgradeDataAccess->_accessId);
				
				access._type = (access_type_t) upgradeDataAccess->_newAccessType;
				if (upgradeDataAccess->_becomesUnsatisfied) {
					access._satisfied = false;
				}
				
				accumulateStepsTriggeredByTask = upgradeDataAccess->_originatorTaskId;
			} else if (dataAccessBecomesSatisfied != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != dataAccessBecomesSatisfied->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(dataAccessBecomesSatisfied->_accessId);
				
				access._satisfied = true;
				
				accumulateStepsTriggeredByTask = dataAccessBecomesSatisfied->_triggererTaskId;
			} else if (removedDataAccess != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != removedDataAccess->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(removedDataAccess->_accessId);
				
				access._deleted = true;
				
				for (auto previousAccessId : access._previousLinks) {
					access_t &previousAccess = getAccess(previousAccessId);
					auto it = previousAccess._nextLinks.find(removedDataAccess->_accessId);
					assert(it != previousAccess._nextLinks.end());
					it->second._status = link_to_next_t::dead_link_status;
				}
				access._previousLinks.clear();
				
				for (auto nextAccessLink : access._nextLinks) {
					// access_t &nextAccess = getAccess(nextAccessLink.first);
					// nextAccess._previousLinks.erase(removedDataAccess->_accessId);
					nextAccessLink.second._status = link_to_next_t::dead_link_status;
				}
				// access._nextLinks.clear();
				
				accumulateStepsTriggeredByTask = removedDataAccess->_triggererTaskId;
			} else if (linkedDataAccess != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != linkedDataAccess->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &sourceAccess = getAccess(linkedDataAccess->_sourceAccessId);
				sourceAccess._nextLinks[linkedDataAccess->_sinkAccessId]._status = link_to_next_t::created_link_status;
				
				accumulateStepsTriggeredByTask = linkedDataAccess->_triggererTaskId;
			} else if (unlinkedDataAccess != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != unlinkedDataAccess->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &sourceAccess = getAccess(unlinkedDataAccess->_sourceAccessId);
				sourceAccess._nextLinks[unlinkedDataAccess->_sinkAccessId]._status = link_to_next_t::dead_link_status;
				// access_t &sinkAccess = getAccess(unlinkedDataAccess->_sinkAccessId);
				// sinkAccess._previousLinks.erase(unlinkedDataAccess->_sourceAccessId);
				
				accumulateStepsTriggeredByTask = unlinkedDataAccess->_triggererTaskId;
			} else {
				assert(false);
			}
		}
		if (accumulateStepsTriggeredByTask != -1) {
			emitFrame(dir, filenameBase, frame);
		}
		
		std::ofstream scriptOS(filenameBase + "-script.sh");
		scriptOS << "#!/bin/sh" << std::endl;
		scriptOS << "set -e" << std::endl;
		scriptOS << std::endl;
		
		scriptOS << "lasttime=0" << std::endl;
		for (int i=0; i < frame; i++) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(8); oss.fill('0'); oss << i;
				stepBase = oss.str();
			}
			scriptOS << "currenttime=$(date +%s)" << std::endl;
			scriptOS << "if [ ${currenttime} -gt ${lasttime} ] ; then" << std::endl;
			scriptOS << "	echo Generating step " << i+1 << "/" << frame << std::endl;
			scriptOS << "	lasttime=${currenttime}" << std::endl;
			scriptOS << "fi" << std::endl;
			scriptOS << "dot -Gfontname=Helvetica -Nfontname=Helvetica -Efontname=Helvetica \"$@\" -Tpdf " << stepBase << ".dot -o " << stepBase << ".pdf" << std::endl;
		}
		scriptOS << std::endl;
		
		scriptOS << "echo Joining into a single file" << std::endl;
		scriptOS << "pdfjoin -q --preamble '\\usepackage{hyperref} \\hypersetup{pdfpagelayout=SinglePage}' --rotateoversize false --outfile " << filenameBase << ".pdf";
		for (int i=0; i < frame; i++) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(8); oss.fill('0'); oss << i;
				stepBase = oss.str();
			}
			scriptOS << " " << stepBase << ".pdf";
		}
		scriptOS << std::endl;
		
		scriptOS << "echo Generated " << filenameBase << ".pdf" << std::endl;
		scriptOS << std::endl;
		
		scriptOS << "echo" << std::endl;
		scriptOS << "echo The contents of " << dir << " can now be safely removed " << std::endl;
		scriptOS << std::endl;
		scriptOS.close();
		
		chmod((filenameBase + "-script.sh").c_str(), S_IRUSR|S_IWUSR|S_IXUSR);
		std::cerr << std::endl << "Generated graph script '" << filenameBase << "-script.sh" << "'" << std::endl;
	}
	
}
