/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"

#include "InstrumentGraph.hpp"
#include "Color.hpp"
#include "ExecutionSteps.hpp"
#include "GenerateEdges.hpp"
#include "PathLength.hpp"
#include "SortAccessGroups.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "system/RuntimeInfo.hpp"

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
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "graph");
	}
	
	
	static long _nextCluster = 1;
	
	static EnvironmentVariable<bool> _shortenFilenames("NANOS6_GRAPH_SHORTEN_FILENAMES", false);
	static EnvironmentVariable<bool> _showSpuriousDependencyStructures("NANOS6_GRAPH_SHOW_SPURIOUS_DEPENDENCY_STRUCTURES", false);
	static EnvironmentVariable<bool> _showDeadDependencyStructures("NANOS6_GRAPH_SHOW_DEAD_DEPENDENCY_STRUCTURES", false);
	static EnvironmentVariable<bool> _showDeadDependencies("NANOS6_GRAPH_SHOW_DEAD_DEPENDENCIES", false);
	static EnvironmentVariable<bool> _showAllSteps("NANOS6_GRAPH_SHOW_ALL_STEPS", false);
	static EnvironmentVariable<bool> _showSuperAccessLinks("NANOS6_GRAPH_SHOW_SUPERACCESS_LINKS", true);
	static EnvironmentVariable<bool> _autoDisplay("NANOS6_GRAPH_DISPLAY", false);
	
	static EnvironmentVariable<std::string> _displayCommand("NANOS6_GRAPH_DISPLAY_COMMAND", "xdg-open");
	
	
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
		assert(taskInfo._nanos6_task_invocation_info != nullptr);
		task_invocation_info_label_map_t::iterator it = _taskInvocationLabel.find(taskInfo._nanos6_task_invocation_info);
		if (it != _taskInvocationLabel.end()) {
			return it->second;
		}
		
		std::string label;
		if (taskInfo._nanos6_task_info->implementations[0].task_label != nullptr) {
			label = taskInfo._nanos6_task_info->implementations[0].task_label;
		} else if ((taskInfo._nanos6_task_invocation_info != nullptr) && (taskInfo._nanos6_task_invocation_info->invocation_source != nullptr)) {
			label = taskInfo._nanos6_task_invocation_info->invocation_source;
		} else if (taskInfo._nanos6_task_info->implementations[0].declaration_source != nullptr) {
			label = taskInfo._nanos6_task_info->implementations[0].declaration_source;
		} else {
			label = std::string();
		}
		
		if (_shortenFilenames) {
			shortenString(label);
		}
		
		_taskInvocationLabel[taskInfo._nanos6_task_invocation_info] = std::move(label);
		return _taskInvocationLabel[taskInfo._nanos6_task_invocation_info];
	}
	
	static std::string indentation;
	
	
	static std::string makeTaskLabel(task_id_t id, task_info_t &taskInfo)
	{
		std::ostringstream oss;
		
		oss << id;
		std::string taskName = getTaskName(taskInfo);
		if (!taskName.empty()) {
			if (taskInfo._phaseList.empty()) {
				oss << "\\n";
			} else {
				oss << ": ";
			}
			oss << taskName;
		}
		
		bool openedSquareBrackets = false;
		if (taskInfo._isIf0) {
			if (!openedSquareBrackets) {
				oss << " [";
				openedSquareBrackets = true;
			} else {
				oss << ", ";
			}
			oss << "if(0)";
		}
		
		if (openedSquareBrackets) {
			oss << "]";
		}
		
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
	
	
	typedef char Bool;
	
	
	
	
	struct dot_linking_labels {
		std::string _nodeLabel;
		std::string _outLabel;
		std::string _inLabel;
	};
	typedef std::map<task_id_t, dot_linking_labels> task_to_dot_linking_labels_t;
	
	
	static task_to_dot_linking_labels_t _taskToDotLinkingLabels;
	
	
	static void emitAccess(
		std::ofstream &ofs, access_t &access
	) {
		std::string color;
		std::ostringstream text;
		
		text << access._id;
		
		if (access._status != Graph::not_created_access_status) {
			text << ": ";
			
			switch (access._type) {
				case READ:
					text << (access.weak() ? "r" : "R");
					color = "#00FF00";
					break;
				case WRITE:
					text << (access.weak() ? "w" : "W");
					color = "#FF0000";
					break;
				case READWRITE:
					text << (access.weak() ? "rw" : "RW");
					color = "#FFB507";
					break;
				case CONCURRENT:
					text << (access.weak() ? "c" : "C");
					color = "#FFFF00";
					break;
				case COMMUTATIVE:
					text << (access.weak() ? "cm" : "CM");
					color = "#FF00FF";
					break;
				case REDUCTION:
					text << (access.weak() ? "red" : "RED");
					color = "#00AFFF";
					break;
				case LOCAL:
					text << "LOC";
					color = "#00FFAF";
					break;
			}
		}
		
		std::string style;
		std::ostringstream colorList;
		switch (access._status) {
			case not_created_access_status:
				color = "#AAAAAA";
				style = "filled";
				colorList << color;
				break;
			case created_access_status:
				style = "filled";
				colorList << color;
				break;
			case removable_access_status:
				style = "filled,dashed";
				colorList << color;
				break;
			case removed_access_status:
				color = "#AAAAAA";
				style = "filled,dotted";
				colorList << color;
				break;
		}
		
		if (_showRegions && (access._status != not_created_access_status)) {
			text << "\\n" << access._accessRegion;
		}
		
		bool haveStatusText = false;
		std::ostringstream statusText;
		
		if (access.satisfied()) {
			haveStatusText = true;
			statusText << "Sat";
		}
		if (access.complete()) {
			haveStatusText = true;
			statusText << "C";
		}
		
		for (std::string const &property : access._otherProperties) {
			haveStatusText = true;
			statusText << property;
		}
		
		if (haveStatusText) {
			text << "\\n" << statusText.str() << "+";
		}
		
		std::string shape;
		switch (access._objectType) {
			case regular_access_type:
				shape = "ellipse";
				break;
			case entry_fragment_type:
				shape = "invhouse";
				break;
			case taskwait_type:
				shape = "octagon";
				break;
			case top_level_sink_type:
				shape = "house";
				break;
		}
		
		ofs << indentation
			<< "data_access_" << access._id
			<< "["
				<< " shape=" << shape
				<< " style=\"" << style << "\""
				<< " label=\"" << text.str() << "\""
				<< " fillcolor=\"" << colorList.str() << "\"";
		if (!access.satisfied()) {
			ofs
				<< " penwidth=2 ";
		} else {
			ofs
				<< " penwidth=1 ";
		}
		ofs
			<< "]"
#ifndef NDEBUG
			<< "\t// " << __FILE__ << ":" << __LINE__
#endif
			<< std::endl;
	}
	
	
	// Returns the id of an emited access in the group
	static data_access_id_t emitAccessGroup(
		std::ofstream &ofs, access_t &firstAccess
	) {
		std::string initialIndentation = indentation;
		indentation = initialIndentation + "\t";
		
		data_access_id_t lastId;
		data_access_id_t firstEmittedId;
		data_access_id_t currentId = firstAccess._id;
		
		bool started = false;
		while (currentId != data_access_id_t()) {
			access_t *access = _accessIdToAccessMap[currentId];
			assert(access != nullptr);
			
			// Skip irrelevant accesses if not explicitly requested
			if (!_showDeadDependencyStructures && (access->_status != created_access_status)) {
				currentId = access->_nextGroupAccess;
				continue;
			}
			
			if (!started) {
				started = true;
				ofs << initialIndentation << "{"
				#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
				#endif
					<< std::endl;
				ofs << indentation << "rank=same;" << std::endl;
			}
			
			emitAccess(ofs, *access);
			if (lastId != data_access_id_t()) {
				ofs << indentation
				<< "data_access_" << lastId
				<< " -> "
				<< "data_access_" << access->_id
				<< " [style=invis]"
#ifndef NDEBUG
				<< "\t// " << __FILE__ << ":" << __LINE__
#endif
				<< std::endl;
			}
			if (firstEmittedId == data_access_id_t()) {
				firstEmittedId = access->_id;
			}
			
			lastId = access->_id;
			currentId = access->_nextGroupAccess;
		}
		
		indentation = initialIndentation;
		
		if (started) {
			ofs << indentation << "}" << std::endl;
		}
		
		return firstEmittedId;
	}
	
	
	static void emitTask(
		std::ofstream &ofs, task_id_t taskId,
		std::ostringstream &linksStream
	) {
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		dot_linking_labels &taskLinkingLabels = _taskToDotLinkingLabels[taskId];
		
		if (!isComposite(taskId)) {
			// A leaf task
			std::ostringstream oss;
			oss << "task" << taskId;
			taskLinkingLabels._nodeLabel = oss.str();
			
			ofs
				<< indentation << taskLinkingLabels._nodeLabel
				<< " [ label=\"" << makeTaskLabel(taskId, taskInfo) << "\" "
				<< getTaskAttributes(taskInfo._status, taskInfo._lastCPU);
			
			if (taskInfo._isIf0) {
				ofs << " shape=\"doubleoctagon\" ";
			} else {
				ofs << " shape=\"box\" ";
			}
			
			ofs << " ]" << std::endl;
			
			taskLinkingLabels._inLabel = taskLinkingLabels._nodeLabel;
			taskLinkingLabels._outLabel = taskLinkingLabels._nodeLabel;
		} else {
			bool sourceLinkIsSet = false;
			
			{
				std::ostringstream oss;
				oss << "cluster_task" << taskId;
				taskLinkingLabels._nodeLabel = oss.str();
			}
			
			std::string initialIndentation = indentation;
			
			// This external and invisible cluster is here just to correct the vertical spacing
			ofs << indentation << "subgraph " << taskLinkingLabels._nodeLabel << "_wrapper {" << std::endl;
			indentation = initialIndentation + "\t";
			ofs << indentation << "style=invis;" << std::endl;
			
			ofs << indentation << "subgraph " << taskLinkingLabels._nodeLabel << " {"
#ifndef NDEBUG
				<< "\t// " << __FILE__ << ":" << __LINE__
#endif
				<< std::endl;
			indentation = initialIndentation + "\t\t";
			ofs << indentation << "label=\"" << makeTaskLabel(taskId, taskInfo) << "\";" << std::endl;
			ofs << indentation << "compound=true;" << std::endl;
			ofs << indentation << "color=\"black\";" << std::endl;
			ofs << indentation << "penwidth=1 ;" << std::endl;
			ofs << indentation << getTaskAttributes(taskInfo._status, taskInfo._lastCPU) << std::endl;
			
			std::vector<std::string> previousPhaseLinks;
			std::vector<std::string> previousPhaseSinkLinks;
			std::vector<Bool> previousPhaseBestBottomLinks;
			std::vector<Bool> previousPhaseElementWithoutSuccessors;
			std::vector<task_status_t> previousPhaseStatuses;
			for (unsigned int phase = 0; phase < taskInfo._phaseList.size(); phase++) {
				phase_t *currentPhase = taskInfo._phaseList[phase];
				
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(currentPhase);
				taskwait_t *taskWait = dynamic_cast<taskwait_t *>(currentPhase);
				
				size_t phaseElements = 0;
				if (taskGroup != nullptr) {
					for (task_id_t childId : taskGroup->_children) {
						task_info_t &childInfo = _taskToInfoMap[childId];
						
						if (!childInfo._isIf0) {
							phaseElements++;
						}
					}
				} else {
					phaseElements = 1;
				}
				
				std::vector<std::string> currentPhaseLinks(phaseElements);
				std::vector<std::string> currentPhaseSourceLinks(phaseElements);
				std::vector<data_access_id_t> currentPhaseTopAccesses;
				std::vector<std::string> currentPhaseSinkLinks(phaseElements);
				std::vector<Bool> bestTopLink(phaseElements, true);
				std::vector<Bool> bestBottomLink(phaseElements, true);
				std::vector<Bool> currentPhaseElementWithoutPredecessors(phaseElements, true);
				std::vector<Bool> currentPhaseElementWithoutSuccessors(phaseElements, true);
				std::vector<task_status_t> currentPhaseStatuses(phaseElements);
				if (taskGroup != nullptr) {
					long currentCluster = _nextCluster++;
					
					{
						std::ostringstream oss;
						oss << "cluster_phase" << currentCluster;
						std::string currentPhaseLink = oss.str();
						
						ofs << indentation << "subgraph " << currentPhaseLink << " {"
#ifndef NDEBUG
							<< "\t// " << __FILE__ << ":" << __LINE__
#endif
							<< std::endl;
					}
					indentation = initialIndentation + "\t\t";
					ofs << indentation << "label=\"\";" << std::endl;
					ofs << indentation << "rank=same;" << std::endl;
					ofs << indentation << "compound=true;" << std::endl;
					ofs << indentation << "style=\"invisible\";" << std::endl;
					
					
					// Emit the access fragments and the taskwait fragments
					if (_showDependencyStructures) {
						for (access_fragment_t *fragment : taskGroup->_allFragments) {
							assert(fragment != nullptr);
							
							// Work at the access group level
							if (fragment->_firstGroupAccess != fragment->_id) {
								continue;
							}
							
							data_access_id_t representativeFragmentId = emitAccessGroup(ofs, *fragment);
							
							data_access_id_t currentFragmentId = representativeFragmentId;
							while (currentFragmentId != data_access_id_t()) {
								access_fragment_t *accessGroupFragment = (access_fragment_t *) _accessIdToAccessMap[currentFragmentId];
								
								if (_showDeadDependencyStructures || (accessGroupFragment->_status == created_access_status)) {
									currentPhaseTopAccesses.push_back(currentFragmentId);
									
									if (!sourceLinkIsSet) {
										for (auto &nextIdAndLink : accessGroupFragment->_nextLinks) {
											link_to_next_t &link = nextIdAndLink.second;
											
											if (taskGroup->_longestPathFirstTaskId == nextIdAndLink.first) {
												task_info_t &nextTaskInfo = _taskToInfoMap[nextIdAndLink.first];
												
												if (_showDeadDependencyStructures) {
													std::ostringstream oss;
													oss << "data_access_" << accessGroupFragment->_id;
													taskLinkingLabels._inLabel = oss.str();
													sourceLinkIsSet = true;
												} else if (link._status == created_link_status) {
													nextTaskInfo._liveAccesses.processIntersecting(
														accessGroupFragment->_accessRegion,
														[&](task_live_accesses_t::iterator nextAccessPosition) -> bool {
															access_t *nextAccess = nextAccessPosition->_access;
															assert(nextAccess != nullptr);
															
															if (_showDeadDependencyStructures || (nextAccess->_status == created_access_status)) {
																std::ostringstream oss;
																oss << "data_access_" << accessGroupFragment->_id;
																taskLinkingLabels._inLabel = oss.str();
																sourceLinkIsSet = true;
																
																return false;
															}
															
															return true;
														}
													);
												}
											}
											
											if (sourceLinkIsSet) {
												break;
											}
										}
									}
								}
								
								currentFragmentId = accessGroupFragment->_nextGroupAccess;
							}
						} // For each access fragment
						
						for (taskwait_fragment_t *fragment : taskGroup->_allTaskwaitFragments) {
							assert(fragment != nullptr);
							
							// Work at the access group level
							if (fragment->_firstGroupAccess != fragment->_id) {
								continue;
							}
							
							emitAccessGroup(ofs, *fragment);
						} // For each taskwait fragment
					} // If must show dependency structures
					
					
					std::map<task_id_t, size_t> taskId2Index;
					{
						size_t index = 0;
						for (task_id_t childId : taskGroup->_children) {
							emitTask(
								ofs, childId,
								linksStream
							);
							
							dot_linking_labels &childLinkingLabels = _taskToDotLinkingLabels[childId];
							task_info_t &childInfo = _taskToInfoMap[childId];
							
							if (!childInfo._isIf0) {
								currentPhaseLinks[index] = childLinkingLabels._nodeLabel;
								currentPhaseSourceLinks[index] = childLinkingLabels._inLabel;
								currentPhaseSinkLinks[index] = childLinkingLabels._outLabel;
								
								bestTopLink[index] = (taskGroup->_longestPathFirstTaskId == childId);
								bestBottomLink[index] = (taskGroup->_longestPathLastTaskId == childId);
								
								taskId2Index[childId] = index;
								
								currentPhaseStatuses[index] = childInfo._status;
								
								currentPhaseElementWithoutPredecessors[index] = !childInfo._hasPredecessorsInSameLevel;
								currentPhaseElementWithoutSuccessors[index] = !childInfo._hasSuccessorsInSameLevel;
							}
							
							// Emit nodes for the accesses
							if (_showDependencyStructures) {
								for (access_t *access : childInfo._allAccesses) {
									assert(access != nullptr);
									assert(access->_objectType == regular_access_type);
									
									if (_showDeadDependencyStructures || (access->_status == created_access_status)) {
										if (!access->hasPrevious()) {
											currentPhaseTopAccesses.push_back(access->_id);
										}
									}
									
									// Work at the access group level
									if (access->_firstGroupAccess != access->_id) {
										continue;
									}
									
									data_access_id_t representativeAccessId = emitAccessGroup(ofs, *access);
									if (representativeAccessId == data_access_id_t()) {
										// Did not actually emit anything
										continue;
									}
									
									if (!sourceLinkIsSet && (taskGroup->_longestPathFirstTaskId == childId)) {
										std::ostringstream oss;
										oss << "data_access_" << representativeAccessId;
										taskLinkingLabels._inLabel = oss.str();
										sourceLinkIsSet = true;
									}
								} // For each access of the child
							} // If must show dependency structures
							
							if (!childInfo._isIf0) {
								index++;
							}
						}
					}
					
					// Check if there are taskwait fragments so that they can be used as phase sinks
					// and link to the actual taskwait
					if (_showDependencyStructures && (taskGroup->_nextTaskwaitId != taskwait_id_t())) {
						taskwait_t *taskwait = _taskwaitToInfoMap[taskGroup->_nextTaskwaitId];
						assert(taskwait != nullptr);
						
						if (_showDeadDependencyStructures && !taskGroup->_allTaskwaitFragments.empty()) {
							for (taskwait_fragment_t *taskwaitFragment : taskGroup->_allTaskwaitFragments) {
								assert(taskwaitFragment != nullptr);
								
								linksStream << "\t"
									<< "data_access_" << taskwaitFragment->_id
									<< " -> "
									<< "taskwait" << taskwait->_taskwaitId
									<< " [";
								if (
									(taskwait->_status == not_created_status)
									|| (taskwait->_status == finished_status)
									|| (taskwait->_status == deleted_status)
									|| (taskwaitFragment->_status == not_created_access_status)
									|| (taskwaitFragment->_status == removed_access_status)
								) {
									if (!_showDeadDependencyStructures) {
										linksStream << " style=invis";
									} else {
										linksStream << " style=dotted"
										<< " color=\"#AAAAAA\""
										<< " fillcolor=\"#AAAAAA\"";
									}
								} else {
									linksStream << " style=dashed"
									<< " color=\"#000000\""
									<< " fillcolor=\"#000000\"";
								}
								linksStream << " ]"
#ifndef NDEBUG
									<< "\t// " << __FILE__ << ":" << __LINE__
#endif
									<< std::endl;
							}
						}
					}
					
					indentation = initialIndentation + "\t";
					ofs << indentation << "}" << std::endl;
				} else if (taskWait != nullptr) {
					if (taskWait->_if0Task == task_id_t()) {
						std::ostringstream oss;
						oss << "taskwait" << taskWait->_taskwaitId;
						std::string currentPhaseLink = oss.str();
						
						ofs
							<< indentation << currentPhaseLink
							<< " [ label=\"" << makeTaskwaitLabel(taskWait->_taskwaitSource) << "\" "
							<< getTaskAttributes(taskWait->_status, taskWait->_lastCPU)
							<< " shape=\"doubleoctagon\" "
							<< " ]"
#ifndef NDEBUG
							<< "\t// " << __FILE__ << ":" << __LINE__
#endif
							<< std::endl;
						
						currentPhaseLinks[0] = currentPhaseLink;
						currentPhaseSourceLinks[0] = currentPhaseLink;
						currentPhaseSinkLinks[0] = currentPhaseLink;
						currentPhaseStatuses[0] = taskWait->_status;
						bestTopLink[0] = true;
						bestBottomLink[0] = true;
					} else {
						// An if(0)'ed task
						dot_linking_labels &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskWait->_if0Task];
						task_info_t const &if0Task = _taskToInfoMap[taskWait->_if0Task];
						
						currentPhaseLinks[0] = if0TaskLinkingLabels._nodeLabel;
						currentPhaseSourceLinks[0] = if0TaskLinkingLabels._outLabel;
						currentPhaseSinkLinks[0] = if0TaskLinkingLabels._inLabel;
						currentPhaseStatuses[0] = if0Task._status;
						
						bestTopLink[0] = true;
						bestBottomLink[0] = true;
					}
				} else {
					assert(false);
				}
				
				if ((phase == 0) && !sourceLinkIsSet) {
					std::vector<std::string> validSourceLinks;
					for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
						if (bestTopLink[currentIndex]) {
							validSourceLinks.push_back(currentPhaseSourceLinks[currentIndex]);
						}
					}
					
					taskLinkingLabels._inLabel = validSourceLinks[validSourceLinks.size()/2];
					sourceLinkIsSet = true;
				}
				
				if (phase == taskInfo._phaseList.size()-1) {
					std::vector<std::string> validSinkLinks;
					for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
						if (bestBottomLink[currentIndex]) {
							validSinkLinks.push_back(currentPhaseSinkLinks[currentIndex]);
						}
					}
					
					taskLinkingLabels._outLabel = validSinkLinks[validSinkLinks.size()/2];
				}
				
				if (!previousPhaseLinks.empty()) {
					size_t previousPhaseElements = previousPhaseLinks.size();
					
					for (size_t previousIndex=0; previousIndex < previousPhaseElements; previousIndex++) {
						if (!previousPhaseElementWithoutSuccessors[previousIndex]) {
							// Link only leaf nodes of the previous phase
							continue;
						}
						
						if (!currentPhaseTopAccesses.empty()) {
							for (data_access_id_t accessId : currentPhaseTopAccesses) {
								linksStream
									<< "\t" << previousPhaseSinkLinks[previousIndex] << " -> data_access_" << accessId << " [ style=\"invis\" ";
								
								if (previousPhaseLinks[previousIndex] != previousPhaseSinkLinks[previousIndex]) {
									linksStream << " ltail=\"" << previousPhaseLinks[previousIndex] << "\" ";
								}
								
								linksStream << " ];"
#ifndef NDEBUG
									<< "\t// " << __FILE__ << ":" << __LINE__
#endif
									<< std::endl; 
							}
							currentPhaseTopAccesses.clear();
						}
					}
				}
				
				previousPhaseLinks = std::move(currentPhaseLinks);
				previousPhaseSinkLinks = std::move(currentPhaseSinkLinks);
				previousPhaseBestBottomLinks = std::move(bestBottomLink);
				previousPhaseElementWithoutSuccessors = std::move(currentPhaseElementWithoutSuccessors);
				previousPhaseStatuses = std::move(currentPhaseStatuses);
			}
			
			// End of cluster
			indentation = initialIndentation + "\t";
			ofs << indentation << "}" << std::endl;
			
			// End of invisible cluster
			indentation = initialIndentation;
			ofs << indentation << "}" << std::endl;
		}
	}
	
	
	static void emitAccessLinksToNext(access_t &access, std::ofstream &ofs)
	{
		dot_linking_labels const &sourceLinkingLabels = _taskToDotLinkingLabels[access._originator];
		
		foreachItersectingNextOfAccess(
			&access,
			[&](access_t &nextAccess, link_to_next_t &linkToNext, __attribute__((unused)) task_info_t &nextTaskInfo) -> bool
			{
				if (!_showDeadDependencyStructures && (nextAccess._status != created_access_status)) {
					return true;
				}
				
				ofs << "\t"
					<< "data_access_" << access._id
					<< " -> "
					<< "data_access_" << nextAccess._id
					<< " [";
				if ((linkToNext._status == not_created_link_status) || (linkToNext._status == dead_link_status)) {
					ofs << " style=dotted"
					<< " color=\"#AAAAAA\""
					<< " fillcolor=\"#AAAAAA\"";
				} else if (
					(access._status == not_created_access_status)
					|| (access._status == removed_access_status)
					|| (nextAccess._status == not_created_access_status)
					|| (nextAccess._status == removed_access_status)
				) {
					ofs << " style=dotted"
					<< " color=\"#AAAAAA\""
					<< " fillcolor=\"#AAAAAA\"";
				} else if (!linkToNext._direct) {
					ofs << " arrowhead=\"vee\""
					<< " style=dotted"
					<< " color=\"#000000\""
					<< " fillcolor=\"#000000\"";
				} else {
					ofs << " style=dashed"
					<< " color=\"#000000\""
					<< " fillcolor=\"#000000\"";
				}
				ofs << " ]"
#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
#endif
					<< std::endl;
				
				if (access._objectType != entry_fragment_type) {
					ofs << "\t" << sourceLinkingLabels._outLabel << " -> " << "data_access_" << nextAccess._id;
					ofs << " [";
					if (sourceLinkingLabels._outLabel != sourceLinkingLabels._nodeLabel) {
						ofs << " ltail=\"" << sourceLinkingLabels._nodeLabel << "\"";
					}
					ofs << " style=\"invis\"";
					ofs << " ]"
#ifndef NDEBUG
						<< "\t// " << __FILE__ << ":" << __LINE__
#endif
						<< std::endl;
				}
				
				return true;
			}
		);
		
		
	}
	
	
	static void emitEdges(std::ofstream &ofs)
	{
		for (auto &taskIdAndInfo : _taskToInfoMap) {
			task_id_t taskId = taskIdAndInfo.first;
			task_info_t &taskInfo = taskIdAndInfo.second;
			dot_linking_labels const &sourceLinkingLabels = _taskToDotLinkingLabels[taskId];
			
			// Task dependencies
			for (auto &successorAndDependencyEdge : taskInfo._outputEdges) {
				task_id_t successorId = successorAndDependencyEdge.first;
				dependency_edge_t &dependencyEdge = successorAndDependencyEdge.second;
				
				dot_linking_labels const &sinkLinkingLabels = _taskToDotLinkingLabels[successorId];
				
				ofs << "\t" << sourceLinkingLabels._outLabel << " -> " << sinkLinkingLabels._inLabel;
				
				ofs << " [";
				
				if (sourceLinkingLabels._outLabel != sourceLinkingLabels._nodeLabel) {
					ofs << " ltail=\"" << sourceLinkingLabels._nodeLabel << "\"";
				}
				if (sinkLinkingLabels._inLabel != sinkLinkingLabels._nodeLabel) {
					ofs << " lhead=\"" << sinkLinkingLabels._nodeLabel << "\"";
				}
				
				task_info_t &successorInfo = _taskToInfoMap[successorId];
				if (!_showDeadDependencies &&
					(!dependencyEdge._hasBeenMaterialized
					|| ((dependencyEdge._activeStrongContributorLinks == 0) && (dependencyEdge._activeWeakContributorLinks == 0))
					|| (taskInfo._status == not_created_status)
					|| (taskInfo._status == finished_status)
					|| (taskInfo._status == deleted_status)
					|| (successorInfo._status == not_created_status)
					|| (successorInfo._status == finished_status)
					|| (successorInfo._status == deleted_status))
				) {
					ofs << " style=\"invis\"";
				} else if (dependencyEdge._activeStrongContributorLinks == 0) {
					ofs << " style=\"dashed\"";
				}
				
				ofs << " ]"
#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
#endif
					<< std::endl;
				
				if (_showDependencyStructures) {
					if (_showDeadDependencyStructures) {
						for (access_t *nextAccess : successorInfo._allAccesses) {
							assert(nextAccess != nullptr);
							
							ofs << "\t" << sourceLinkingLabels._outLabel << " -> data_access_" << nextAccess->_id << " [ style=\"invis\" ";
							if (sourceLinkingLabels._outLabel != sourceLinkingLabels._nodeLabel) {
								ofs << " ltail=\"" << sourceLinkingLabels._nodeLabel << "\"";
							}
							ofs << " ]"
#ifndef NDEBUG
								<< "\t// " << __FILE__ << ":" << __LINE__
#endif
								<< std::endl;
						}
					} else {
						successorInfo._liveAccesses.processAll(
							[&](task_live_accesses_t::iterator position) -> bool {
								access_t *nextAccess = position->_access;
								assert(nextAccess != nullptr);
								
								if (nextAccess->_status != created_access_status) {
									return true;
								}
								
								ofs << "\t" << sourceLinkingLabels._outLabel << " -> data_access_" << nextAccess->_id << " [ style=\"invis\" ";
								if (sourceLinkingLabels._outLabel != sourceLinkingLabels._nodeLabel) {
									ofs << " ltail=\"" << sourceLinkingLabels._nodeLabel << "\"";
								}
								ofs << " ]"
#ifndef NDEBUG
									<< "\t// " << __FILE__ << ":" << __LINE__
#endif
									<< std::endl;
								
								return true;
							}
						);
					}
				}
			}
			
			if (taskInfo._precedingTaskwait != taskwait_id_t()) {
				taskwait_t *taskwait = _taskwaitToInfoMap[taskInfo._precedingTaskwait];
				assert(taskwait != nullptr);
				
				ofs << "\t";
				if (taskwait->_if0Task != task_id_t()) {
					dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
					ofs << if0TaskLinkingLabels._outLabel;
				} else {
					ofs << "taskwait" << taskInfo._precedingTaskwait;
				}
				
				ofs << " -> " << sourceLinkingLabels._inLabel;
				ofs << " [";
			
				if (taskwait->_if0Task != task_id_t()) {
					dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
					if (if0TaskLinkingLabels._nodeLabel != if0TaskLinkingLabels._outLabel) {
						ofs << " ltail=\"" << if0TaskLinkingLabels._nodeLabel << "\"";
					}
				}
				
				if (sourceLinkingLabels._inLabel != sourceLinkingLabels._nodeLabel) {
					ofs << " lhead=\"" << sourceLinkingLabels._nodeLabel << "\"";
				}
				
				if (
					(taskInfo._status == not_created_status)
					|| (taskInfo._status == finished_status)
					|| (taskInfo._status == deleted_status)
					|| (taskwait->_status == not_created_status)
					|| (taskwait->_status == finished_status)
					|| (taskwait->_status == deleted_status)
				) {
					if (_showDeadDependencies) {
						ofs << " style=\"dashed\"";
					} else {
						ofs << " style=\"invis\"";
					}
				}
				
				ofs << " ]"
#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
#endif
					<< std::endl;
			}
			
			if (taskInfo._succedingTaskwait != taskwait_id_t()) {
				taskwait_t *taskwait = _taskwaitToInfoMap[taskInfo._succedingTaskwait];
				assert(taskwait != nullptr);
				
				ofs << "\t" << sourceLinkingLabels._outLabel << " -> ";
				if (taskwait->_if0Task != task_id_t()) {
					dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
					ofs << if0TaskLinkingLabels._inLabel;
				} else {
					ofs << "taskwait" << taskInfo._succedingTaskwait;
				}
				
				ofs << " [";
				
				if (sourceLinkingLabels._outLabel != sourceLinkingLabels._nodeLabel) {
					ofs << " ltail=\"" << sourceLinkingLabels._nodeLabel << "\"";
				}
				
				if (taskwait->_if0Task != task_id_t()) {
					dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
					if (if0TaskLinkingLabels._nodeLabel != if0TaskLinkingLabels._inLabel) {
						ofs << " lhead=\"" << if0TaskLinkingLabels._nodeLabel << "\"";
					}
				}
				
				if (
					(taskInfo._status == not_created_status)
					|| (taskInfo._status == finished_status)
					|| (taskInfo._status == deleted_status)
					|| (taskwait->_status == not_created_status)
					|| (taskwait->_status == finished_status)
					|| (taskwait->_status == deleted_status)
				) {
					if (_showDeadDependencies) {
						ofs << " style=\"dashed\"";
					} else {
						ofs << " style=\"invis\"";
					}
				}
				
				ofs << " ]"
#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
#endif
					<< std::endl;
			}
			
			if (!_showDependencyStructures) {
				continue;
			}
			
			// Access links
			for (access_t *access : taskInfo._allAccesses) {
				assert(access != nullptr);
				
				// Skip irrelevant accesses if not explicitly requested
				if (!_showDeadDependencyStructures && (access->_status != created_access_status)) {
					continue;
				}
				
				// Link to superaccess
				if (_showSuperAccessLinks) {
					if ((access->_superAccess != data_access_id_t()) && (access->_status != not_created_access_status)) {
						access_t *superAccess = _accessIdToAccessMap[access->_superAccess];
						assert(superAccess != nullptr);
						
						bool isFirst = !access->hasPrevious();
						bool isLast = access->_nextLinks.empty();
						if (isFirst || isLast) {
							if (_showDeadDependencyStructures || (superAccess->_status != removed_access_status)) {
								std::ostringstream arrowType;
								
								if (isFirst) {
									arrowType << "diamond";
								}
								if (isLast) {
									arrowType << "odiamond";
								}
								
								ofs << "\t" 
									<< "data_access_" << access->_superAccess
									<< " -> "
									<< "data_access_" << access->_id
									<< " ["
										<< " dir=\"both\""
										<< " arrowhead=\"empty\""
										<< " arrowtail=\"" << arrowType.str() << "\""
										<< " style=dotted"
										<< " color=\"#888888\""
										<< " fillcolor=\"#888888\""
									<< "]"
#ifndef NDEBUG
									<< "\t// " << __FILE__ << ":" << __LINE__
#endif
									<< std::endl;
							}
						}
					}
				}
				
				// Link accesses to their originator
				assert(access->_originator != task_id_t());
				dot_linking_labels &taskLinkingLabels = _taskToDotLinkingLabels[access->_originator];
				ofs << "\t"
					<< "data_access_" << access->_id
					<< " -> "
					<< taskLinkingLabels._inLabel
					<< " [";
				if (taskLinkingLabels._nodeLabel != taskLinkingLabels._inLabel) {
					ofs << " lhead=\"" << taskLinkingLabels._nodeLabel << "\"";
				}
				if ((access->_status == not_created_access_status) || (access->_status == removed_access_status)) {
					ofs << " style=dotted color=\"#AAAAAA\" fillcolor=\"#AAAAAA\"";
				} else if (access->weak() || access->satisfied()) {
					ofs << " style=dashed color=\"#888888\" fillcolor=\"#888888\"";
				} else {
					ofs << " style=dashed color=\"#000000\" fillcolor=\"#000000\"";
				}
				ofs << " ]"
#ifndef NDEBUG
					<< "\t// " << __FILE__ << ":" << __LINE__
#endif
					<< std::endl;
				
				// Links to next
				emitAccessLinksToNext(*access, ofs);
			} // For each access
			
			// Emit links from accesses to fragments and from fragments to subaccesses
			for (phase_t *phase : taskInfo._phaseList) {
				assert(phase != nullptr);
				
				task_group_t *taskGroup = dynamic_cast<task_group_t *>(phase);
				if (taskGroup == nullptr) {
					continue;
				}
				assert(taskGroup != nullptr);
				
				for (access_fragment_t *fragment : taskGroup->_allFragments) {
					assert(fragment != nullptr);
					
					// Skip irrelevant fragments if not explicitly requested
					if (!_showDeadDependencyStructures && (fragment->_status != created_access_status)) {
						continue;
					}
					
					// From superaccess to fragment
					if (_showSuperAccessLinks) {
						for (access_t *superAccess : taskInfo._allAccesses) {
							assert(superAccess != nullptr);
							
							if (superAccess->_accessRegion.intersect(fragment->_accessRegion).empty()) {
								continue;
							}
							
							// Skip irrelevant superaccesses if not explicitly requested
							if (!_showDeadDependencyStructures && (superAccess->_status != created_access_status)) {
								continue;
							}
							
							bool isLast = fragment->_nextLinks.empty();
							std::string arrowType;
							
							if (!isLast) {
								arrowType = "diamond";
							} else {
								arrowType = "diamondodiamond";
							}
							
							ofs << "\t" 
								<< "data_access_" << superAccess->_id
								<< " -> "
								<< "data_access_" << fragment->_id
								<< " ["
								<< " dir=\"both\""
								<< " arrowhead=\"empty\""
								<< " arrowtail=\"" << arrowType << "\"";
							if ((superAccess->_status != created_access_status) || (fragment->_status != created_access_status)) {
								ofs
									<< " style=dotted"
									<< " color=\"#AAAAAA\""
									<< " fillcolor=\"#AAAAAA\"";
							} else {
								ofs
									<< " style=dotted"
									<< " color=\"#888888\""
									<< " fillcolor=\"#888888\"";
							}
							ofs
								<< "]"
	#ifndef NDEBUG
								<< "\t// " << __FILE__ << ":" << __LINE__
	#endif
								<< std::endl;
						}
					}
					
					// From fragment to next
					emitAccessLinksToNext(*fragment, ofs);
				}
				
			} // For each task group
			
		} // For each task
		
		for (auto &taskwaitIdAndTaskwait : _taskwaitToInfoMap) {
			taskwait_t *taskwait = taskwaitIdAndTaskwait.second;
			assert(taskwait != nullptr);
			
			if (taskwait->_immediateNextTaskwait == taskwait_id_t()) {
				continue;
			}
			taskwait_t *immediateNextTaskwait = _taskwaitToInfoMap[taskwait->_immediateNextTaskwait];
			assert(immediateNextTaskwait != nullptr);
			
			ofs << "\t";
			if (taskwait->_if0Task != task_id_t()) {
				dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
				ofs << if0TaskLinkingLabels._inLabel;
			} else {
				ofs << "taskwait" << taskwait->_taskwaitId;
			}
			
			ofs << " -> ";
			
			if (immediateNextTaskwait->_if0Task != task_id_t()) {
				dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[immediateNextTaskwait->_if0Task];
				ofs << if0TaskLinkingLabels._inLabel;
			} else {
				ofs << "taskwait" << immediateNextTaskwait->_taskwaitId;
			}
			
			ofs << " [";
			
			if (taskwait->_if0Task != task_id_t()) {
				dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[taskwait->_if0Task];
				if (if0TaskLinkingLabels._nodeLabel != if0TaskLinkingLabels._inLabel) {
					ofs << " ltail=\"" << if0TaskLinkingLabels._nodeLabel << "\"";
				}
			}
			
			if (immediateNextTaskwait->_if0Task != task_id_t()) {
				dot_linking_labels const &if0TaskLinkingLabels = _taskToDotLinkingLabels[immediateNextTaskwait->_if0Task];
				if (if0TaskLinkingLabels._nodeLabel != if0TaskLinkingLabels._inLabel) {
					ofs << " lhead=\"" << if0TaskLinkingLabels._nodeLabel << "\"";
				}
			}
			
			if (
				(immediateNextTaskwait->_status == not_created_status)
				|| (immediateNextTaskwait->_status == finished_status)
				|| (immediateNextTaskwait->_status == deleted_status)
				|| (taskwait->_status == not_created_status)
				|| (taskwait->_status == finished_status)
				|| (taskwait->_status == deleted_status)
			) {
				if (_showDeadDependencies) {
					ofs << " style=\"dashed\"";
				} else {
					ofs << " style=\"invis\"";
				}
			}
			
			ofs << " ]"
#ifndef NDEBUG
				<< "\t// " << __FILE__ << ":" << __LINE__
#endif
				<< std::endl;
		}
	}
	
	
	typedef std::list<std::string> log_t;
	static log_t currentLog;
	
	
	static void emitLog(std::ofstream &ofs, std::ofstream *logFile)
	{
		static int step = 1;
		
		if (!currentLog.empty()) {
			ofs << indentation << "{ rank=max;" << std::endl;
			ofs << indentation << "\tlogbox [ shape=plaintext label=<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" cellpadding=\"4\">" << std::endl;
			
			for (std::string &currentEntry : currentLog) {
				ofs << indentation << "\t\t<tr><td align=\"right\">" << step << "</td><td align=\"left\">" << currentEntry << "</td></tr>" << std::endl;
				if (logFile != nullptr) {
					(*logFile) << "\t" << step << "\t" << currentEntry << std::endl;
				}
				step++;
			}
			
			ofs << indentation << "\t\t</table>" << std::endl;
			ofs << indentation << "\t> ]" << std::endl;
			ofs << indentation << "}" << std::endl;
		}
	}
	
	
	static void dumpGraph(std::ofstream &ofs, std::ofstream *logFile)
	{
		indentation = "";
		ofs << "digraph {" << std::endl;
		indentation = "\t";
		ofs << indentation << "compound=true;" << std::endl;
		ofs << indentation << "nanos6_start [shape=Mdiamond];" << std::endl;
		ofs << indentation << "nanos6_end [shape=Msquare];" << std::endl;
		
		std::ostringstream linksStream;
		emitTask(ofs, 0, linksStream);
		dot_linking_labels &mainTaskLinkingLabels = _taskToDotLinkingLabels[0];
		
		ofs << indentation << "nanos6_start -> " << mainTaskLinkingLabels._inLabel;
		if (mainTaskLinkingLabels._inLabel != mainTaskLinkingLabels._nodeLabel) {
			ofs << "[ lhead=\"" << mainTaskLinkingLabels._nodeLabel << "\" ]";
		}
		ofs << ";" << std::endl;
		
		ofs << indentation << mainTaskLinkingLabels._outLabel << " -> nanos6_end";
		if (mainTaskLinkingLabels._outLabel != mainTaskLinkingLabels._nodeLabel) {
			ofs << " [ ltail=\"" << mainTaskLinkingLabels._nodeLabel << "\" ]";
		}
		ofs << ";" << std::endl;
		
		ofs << linksStream.str();
		
		emitEdges(ofs);
		
		if (_showLog) {
			emitLog(ofs, logFile);
		}
		
		ofs << "}" << std::endl;
		indentation = "";
	}
	
	
	static void emitFrame(std::string const &dir, std::string const &filenameBase, int &frame, std::ofstream *logFile)
	{
		// To avoid spurious differences between frames
		_nextCluster = 1;
		
		// Emit a graph frame
		std::ostringstream oss;
		oss << dir << "/" << filenameBase << "-step";
		oss.width(8); oss.fill('0'); oss << frame;
		oss.width(0); oss.fill(' '); oss << ".dot";
		
		if (logFile != nullptr) {
			(*logFile) << "Page " << frame+1 << ": [" << oss.str() << "]" << std::endl;
		}
		
		std::ofstream ofs(oss.str());
		dumpGraph(ofs, logFile);
		ofs.close();
		
		if (logFile != nullptr) {
			(*logFile) << std::endl;
		}
		
		frame++;
		currentLog.clear();
	}
	
	
	void shutdown()
	{
		std::string filenameBase;
		std::string logName;
		{
			struct timeval tv;
			gettimeofday(&tv, nullptr);
			
			std::ostringstream oss;
			oss << "graph-"
#if HAVE_GETHOSTID
				<< gethostid() << "-"
#endif
				<< getpid() << "-" << tv.tv_sec;
			filenameBase = oss.str();
			
			std::ostringstream oss2;
			oss2 << "log-"
#if HAVE_GETHOSTID
				<< gethostid() << "-"
#endif
				<< getpid() << "-" << tv.tv_sec << ".txt";
			logName = oss2.str();
		}
		
		std::string dir = filenameBase + std::string("-components");
		int rc = mkdir(dir.c_str(), 0755);
		if (rc != 0) {
			FatalErrorHandler::handle(errno, " trying to create directory '", dir, "'");
		}
		
		// Derive the actual edges from the access links
		generateEdges();
		
		// Find out the fencing relations between tasks and taskwaits
		generateTaskwaitRelations();
		
		// Find the longest dependency paths for layout reasons
		findTopmostTasksAndPathLengths(0);
		
		// Find a fixed order for the accesses and fragments of the same group
		if (_showDependencyStructures) {
			sortAccessGroups();
		} else {
			clearLiveAccessesAndGroups();
		}
		
		std::ofstream *logStream = nullptr;
		if (_showLog) {
			logStream = new std::ofstream(logName);
		}
		
		// Simultation loop
		int frame=0;
		execution_step_flush_state_t flushState;
		for (execution_step_t *executionStep : _executionSequence) {
			assert(executionStep != nullptr);
			
			if (_showAllSteps || executionStep->needsFlushBefore(flushState)) {
				emitFrame(dir, filenameBase, frame, logStream);
			}
			
			executionStep->execute();
			if (_showLog) {
				currentLog.push_back(executionStep->describe());
			}
			
			if (!_showAllSteps && executionStep->needsFlushAfter(flushState)) {
				emitFrame(dir, filenameBase, frame, logStream);
			}
		}
		
		// Flush last frame if necessary
		if (_showAllSteps || !flushState._hasAlreadyFlushed) {
			emitFrame(dir, filenameBase, frame, logStream);
		}
		
		if (_showLog) {
			logStream->close();
			delete logStream;
		}
		
		std::string scriptName = filenameBase + "-script.sh";
		
		std::ofstream scriptOS(scriptName);
		scriptOS << "#!/bin/sh" << std::endl;
		scriptOS << "set -e" << std::endl;
		scriptOS << std::endl;
		
		scriptOS << "if which parallel >&/dev/null ; then" << std::endl;
		scriptOS << "\tls " << dir << "/" << filenameBase << "-step" << "*.dot"
			<< " | parallel --progress --eta dot -Gfontname=Helvetica -Nfontname=Helvetica -Efontname=Helvetica \"$@\" -Tpdf " << "'{}'" << " -o " << "'{.}.pdf'" << std::endl;
		scriptOS << "else" << std::endl;
		
		
		scriptOS << "lasttime=0" << std::endl;
		for (int i=0; i < frame; i++) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(8); oss.fill('0'); oss << i;
				stepBase = oss.str();
			}
			scriptOS << "\tcurrenttime=$(date +%s)" << std::endl;
			scriptOS << "\tif [ ${currenttime} -gt ${lasttime} ] ; then" << std::endl;
			scriptOS << "\t	echo Generating step " << i+1 << "/" << frame << std::endl;
			scriptOS << "\t	lasttime=${currenttime}" << std::endl;
			scriptOS << "\tfi" << std::endl;
			scriptOS << "\tdot -Gfontname=Helvetica -Nfontname=Helvetica -Efontname=Helvetica \"$@\" -Tpdf " << stepBase << ".dot -o " << stepBase << ".pdf" << std::endl;
		}
		scriptOS << "fi" << std::endl;
		
		scriptOS << std::endl;
		
		scriptOS << "echo Joining into a single file" << std::endl;
		
		// pdfunite
		scriptOS << "if which pdfunite >&/dev/null ; then" << std::endl;
		for (int i=0; i < frame; i+=100) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(6); oss.fill('0'); oss << i/100;
				oss << "*.pdf";
				stepBase = oss.str();
			}
			std::string intermediate;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-100steps-";
				oss.width(6); oss.fill('0'); oss << i/100;
				oss << ".pdf";
				intermediate = oss.str();
			}
			
			scriptOS << "\tpdfunite " << stepBase << " " << intermediate << std::endl;  
		}
		scriptOS << "\tpdfunite " << dir << "/" << filenameBase << "-100steps-*.pdf " << filenameBase << ".pdf"
			<< " && rm -f " << dir << "/" << filenameBase << "-100steps-*.pdf" << std::endl;
		
		
		// pdfjoin
		scriptOS << "elif which pdfjoin >&/dev/null ; then" << std::endl;
		
		for (int i=0; i < frame; i+=100) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(6); oss.fill('0'); oss << i/100;
				oss << "*.pdf";
				stepBase = oss.str();
			}
			std::string intermediate;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-100steps-";
				oss.width(6); oss.fill('0'); oss << i/100;
				oss << ".pdf";
				intermediate = oss.str();
			}
			
			scriptOS << "\tpdfjoin -q --preamble '\\usepackage{hyperref} \\hypersetup{pdfpagelayout=SinglePage}' --rotateoversize false --outfile " << intermediate << " " << stepBase << std::endl;  
		}
		scriptOS << "\tpdfjoin -q --preamble '\\usepackage{hyperref} \\hypersetup{pdfpagelayout=SinglePage}' --rotateoversize false --outfile " << filenameBase << ".pdf " << dir << "/" << filenameBase << "-100steps-*.pdf"
		<< " && rm -f " << dir << "/" << filenameBase << "-100steps-*.pdf" << std::endl;
		
		// Without pdfunite nor pdfjoin
		scriptOS << "else" << std::endl;
		
		scriptOS << "\techo 'Warning: did not find neither pdfunite (from poppler) nor pdfjoin (from latex) to collect the individual steps into a single PDF.'" << std::endl;
		scriptOS << "\techo 'Individual PDF files for each step have been left inside the \"" << dir << "\" directory.'" << std::endl;
		scriptOS << "\texit" << std::endl;
		
		scriptOS << "fi" << std::endl;
		
		scriptOS << "echo Generated " << filenameBase << ".pdf" << std::endl;
		scriptOS << std::endl;
		
		scriptOS << "echo" << std::endl;
		scriptOS << "echo The contents of " << dir << " can now be safely removed " << std::endl;
		scriptOS << std::endl;
		scriptOS.close();
		
		chmod(scriptName.c_str(), S_IRUSR|S_IWUSR|S_IXUSR);
		
		if (!_autoDisplay) {
			std::cerr << std::endl << "Generated graph script '" << scriptName << "'" << std::endl;
			return;
		}
		
		{
			std::ostringstream oss;
			oss << "./" << scriptName;
			rc = system(oss.str().c_str());
			if (rc != 0) {
				std::cerr << "Error: Execution of '" << scriptName << "' returned code " << rc << std::endl;
				return;
			}
		}
		
		std::string pdfName = filenameBase + ".pdf";
		std::ostringstream displayCommandLine;
		if (_displayCommand.isPresent()) {
			displayCommandLine << ((std::string) _displayCommand) << " " << pdfName;
		} else {
			displayCommandLine << "xdg-open " << pdfName << " || evince " << pdfName << " || okular " << pdfName << " || acroread " << pdfName;
		}
		rc = system(displayCommandLine.str().c_str());
		if (rc != 0) {
			std::cerr << "Error: Execution of '" << displayCommandLine.str() << "' returned code " << rc << std::endl;
			return;
		}
	}
	
}
