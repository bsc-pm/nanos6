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
		if ((sourceStatus != not_created_status) && (sinkStatus != not_created_status))
		{
			return "";
		} else {
			return " color=\"#888888\" fillcolor=\"#888888\" ";
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
					
					{
						size_t index = 0;
						for (task_id_t childId : taskGroup->_children) {
							emitTask(
								ofs, childId,
								currentPhaseLinks[index],
								currentPhaseSourceLinks[index], currentPhaseSinkLinks[index],
								currentPhaseElementsHavePredecessors[index], currentPhaseElementsHaveSuccessors[index]
							);
							
							task_info_t const &childInfo = _taskToInfoMap[childId];
							currentPhaseStatuses[index] = childInfo._status;
							
							index++;
						}
					}
					
					for (edge_t edge : taskGroup->_dependencyEdges) {
						ofs << indentation << makeTaskNodeName(edge._source) << " -> " << makeTaskNodeName(edge._sink);
						task_info_t const &sourceInfo = _taskToInfoMap[edge._source];
						task_info_t const &sinkInfo = _taskToInfoMap[edge._sink];
						
						if ((sourceInfo._status == not_created_status) || (sourceInfo._status == finished_status)
							|| (sinkInfo._status == not_created_status) || (sinkInfo._status == finished_status))
						{
							ofs << "[  color=\"#888888\" fillcolor=\"#888888\" ]" << std::endl;
						} else {
							ofs << std::endl;
						}
					}
					
					for (auto element : taskGroup->_accessSequences) {
						access_sequence_t &accessSequence = element.second;
						
						if (!_showSpuriousDependencyStructures && (accessSequence._accesses.size() <= 1)) {
							// Skip one element access sequences
							continue;
						}
						
						data_access_id_t lastAccessId = -1;
						for (auto element2 : accessSequence._accesses) {
							data_access_id_t accessId = element2.first;
							access_t &access = element2.second;
							
							if (!_showDeadDependencyStructures && (access._deleted || (access._type == NOT_CREATED))) {
								// Skip dead access sequence segments
								continue;
							}
							
							ofs << indentation << "data_access_" << accessId << "[ shape=ellipse";
							switch (access._type) {
								case READ:
									ofs << " label=\"R\" style=\"filled,dashed\"";
									if (access._deleted) {
										ofs << " fillcolor=\"#AAFFAA\"";
									} else {
										ofs << " fillcolor=\"#00FF00\"";
									}
									break;
								case WRITE:
									ofs << " label=\"W\" style=\"filled,dashed\"";
									if (access._deleted) {
										ofs << " fillcolor=\"#FFAAAA\"";
									} else {
										ofs << " fillcolor=\"#FF0000\"";
									}
									break;
								case READWRITE:
									ofs << " label=\"RW\" style=\"filled,dashed\"";
									if (access._deleted) {
										ofs << " fillcolor=\"#EEEE66\"";
									} else {
										ofs << " fillcolor=\"#888800\"";
									}
									break;
								case NOT_CREATED:
									ofs << " label=\"--\" style=\"filled,dashed\" fillcolor=\"#AAAAAA\"";
									assert(!access._deleted);
									break;
							}
							
							if (access._satisfied) {
								ofs << " penwidth=2";
							} else {
								ofs << " penwidth=1";
							}
							ofs << " ]" << std::endl;
							
							assert(access._originator != -1);
							ofs << indentation << "data_access_" << accessId << " -> " << makeTaskNodeName(access._originator);
							if ((access._type == NOT_CREATED) || access._deleted) {
								ofs << "[ style=\"invis\" ]" << std::endl;
							} else if (access._satisfied) {
								ofs << "[ style=dashed color=\"#888888\" fillcolor=\"#888888\" ]" << std::endl;
							} else {
								ofs << "[ style=dashed color=\"#000000\" fillcolor=\"#000000\" ]" << std::endl;
							}
							
							if (lastAccessId != -1) {
								ofs << indentation << "data_access_" << lastAccessId << " -> data_access_" << accessId << " [ weight=8 ]" << std::endl;
							}
							lastAccessId = accessId;
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
				
				if (phase == 0) {
					std::vector<std::string> validSourceLinks;
					for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
						if (!currentPhaseElementsHavePredecessors[currentIndex]) {
							validSourceLinks.push_back(currentPhaseSourceLinks[currentIndex]);
						}
					}
					
					sourceLink = validSourceLinks[validSourceLinks.size()/2];
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
								<< " ];" << std::endl;
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
			ofs << "[ ltail=\"" << mainTaskLink << "\" ]";
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
			
			// Get the kind of step
			create_task_step_t *createTask = dynamic_cast<create_task_step_t *> (executionStep);
			enter_task_step_t *enterTask = dynamic_cast<enter_task_step_t *> (executionStep);
			exit_task_step_t *exitTask = dynamic_cast<exit_task_step_t *> (executionStep);
			
			enter_taskwait_step_t *enterTaskwait = dynamic_cast<enter_taskwait_step_t *> (executionStep);
			exit_taskwait_step_t *exitTaskwait = dynamic_cast<exit_taskwait_step_t *> (executionStep);
			
			enter_usermutex_step_t *enterUsermutex = dynamic_cast<enter_usermutex_step_t *> (executionStep);
			block_on_usermutex_step_t *blockedOnUsermutex = dynamic_cast<block_on_usermutex_step_t *> (executionStep);
			exit_usermutex_step_t *exitUsermutex = dynamic_cast<exit_usermutex_step_t *> (executionStep);
			
			register_task_access_in_sequence_step_t *registerTaskAccessInSequence = dynamic_cast<register_task_access_in_sequence_step_t *> (executionStep);
			upgrade_task_access_in_sequence_step_t *upgradeTaskAccessInSquence = dynamic_cast<upgrade_task_access_in_sequence_step_t *> (executionStep);
			task_access_in_sequence_becomes_satisfied_step_t *taskAccessInSequenceBecomesSatisfied = dynamic_cast<task_access_in_sequence_becomes_satisfied_step_t *> (executionStep);
			removed_task_access_from_sequence_step_t *removedTaskAccessFromSequence = dynamic_cast<removed_task_access_from_sequence_step_t *> (executionStep);
			
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
			} else if (registerTaskAccessInSequence != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != registerTaskAccessInSequence->_originatorTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(
					registerTaskAccessInSequence->_sequenceId,
					registerTaskAccessInSequence->_accessId,
					registerTaskAccessInSequence->_originatorTaskId
				);
				
				access._originator = registerTaskAccessInSequence->_originatorTaskId;
				access._satisfied = registerTaskAccessInSequence->_satisfied;
				access._type = (access_type_t) registerTaskAccessInSequence->_accessType;
				
				accumulateStepsTriggeredByTask = registerTaskAccessInSequence->_originatorTaskId;
			} else if (upgradeTaskAccessInSquence != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != upgradeTaskAccessInSquence->_originatorTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(
					upgradeTaskAccessInSquence->_sequenceId,
					upgradeTaskAccessInSquence->_accessId,
					upgradeTaskAccessInSquence->_originatorTaskId
				);
				
				access._type = (access_type_t) upgradeTaskAccessInSquence->_newAccessType;
				if (upgradeTaskAccessInSquence->_becomesUnsatisfied) {
					access._satisfied = false;
				}
				
				accumulateStepsTriggeredByTask = upgradeTaskAccessInSquence->_originatorTaskId;
			} else if (taskAccessInSequenceBecomesSatisfied != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != taskAccessInSequenceBecomesSatisfied->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(
					taskAccessInSequenceBecomesSatisfied->_sequenceId,
					taskAccessInSequenceBecomesSatisfied->_accessId,
					taskAccessInSequenceBecomesSatisfied->_targetTaskId
				);
				
				access._satisfied = true;
				
				accumulateStepsTriggeredByTask = taskAccessInSequenceBecomesSatisfied->_triggererTaskId;
			} else if (removedTaskAccessFromSequence != nullptr) {
				if ((accumulateStepsTriggeredByTask != -1) && (accumulateStepsTriggeredByTask != removedTaskAccessFromSequence->_triggererTaskId)) {
					emitFrame(dir, filenameBase, frame);
				}
				
				access_t &access = getAccess(
					removedTaskAccessFromSequence->_sequenceId,
					removedTaskAccessFromSequence->_accessId,
					removedTaskAccessFromSequence->_triggererTaskId
				);
				
				access._deleted = true;
				
				accumulateStepsTriggeredByTask = removedTaskAccessFromSequence->_triggererTaskId;
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
