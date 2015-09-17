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
	
	
	static void emitTask(
		std::ofstream &ofs, task_id_t taskId,
		std::string /* OUT */ &taskLink, std::string /* OUT */ &sourceLink, std::string /* OUT */ &sinkLink)
	{
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		
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
					
					size_t subtasks = taskGroup->_children.size();
					for (size_t index=0; index < subtasks; index++) {
						task_id_t childId = taskGroup->_children[index];
						emitTask(ofs, childId, currentPhaseLinks[index], currentPhaseSourceLinks[index], currentPhaseSinkLinks[index]);
						
						task_info_t const &childInfo = _taskToInfoMap[childId];
						currentPhaseStatuses[index] = childInfo._status;
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
				} else {
					assert(false);
				}
				
				if (phase == 0) {
					sourceLink = currentPhaseSourceLinks[phaseElements/2];
				}
				if (phase == taskInfo._phaseList.size()-1) {
					sinkLink = currentPhaseSinkLinks[phaseElements/2];
				}
				
				if (!previousPhaseLinks.empty()) {
					size_t previousPhaseElements = previousPhaseLinks.size();
					
					for (size_t previousIndex=0; previousIndex < previousPhaseElements; previousIndex++) {
						for (size_t currentIndex=0; currentIndex < phaseElements; currentIndex++) {
							
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
		emitTask(ofs, 0, mainTaskLink, mainTaskSourceLink, mainTaskSinkLink);
		
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
		
		int step=0;
		{
			std::ostringstream oss;
			oss << dir << "/" << filenameBase << "-step";
			oss.width(8); oss.fill('0'); oss << step;
			oss.width(0); oss.fill(' '); oss << ".dot";
			
			std::ofstream ofs(oss.str());
			dumpGraph(ofs);
			ofs.close();
		}
		step++;
		
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
			
			// Update the status
			if (createTask != nullptr) {
				task_id_t taskId = createTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == not_created_status);
				taskInfo._status = not_started_status;
				taskInfo._lastCPU = createTask->_cpu;
			} else if (enterTask != nullptr) {
				task_id_t taskId = enterTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert((taskInfo._status == not_started_status) || (taskInfo._status == blocked_status));
				taskInfo._status = started_status;
				taskInfo._lastCPU = enterTask->_cpu;
			} else if (exitTask != nullptr) {
				task_id_t taskId = exitTask->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				taskInfo._status = finished_status;
				taskInfo._lastCPU = exitTask->_cpu;
			} else if (enterTaskwait != nullptr) {
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
				task_id_t taskId = enterUsermutex->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert((taskInfo._status == started_status) || (taskInfo._status == blocked_status));
				taskInfo._status = started_status;
				taskInfo._lastCPU = enterUsermutex->_cpu;
			} else if (blockedOnUsermutex != nullptr) {
				task_id_t taskId = blockedOnUsermutex->_taskId;
				task_info_t &taskInfo = _taskToInfoMap[taskId];
				assert(taskInfo._status == started_status);
				taskInfo._status = started_status;
				taskInfo._lastCPU = blockedOnUsermutex->_cpu;
			} else if (exitUsermutex != nullptr) {
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
			} else {
				assert(false);
			}
			
			// Emit a graph frame
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(8); oss.fill('0'); oss << step;
				oss.width(0); oss.fill(' '); oss << ".dot";
				
				std::ofstream ofs(oss.str());
				dumpGraph(ofs);
				ofs.close();
			}
			step++;
		}
		
		std::ofstream scriptOS(filenameBase + "-script.sh");
		scriptOS << "#!/bin/sh" << std::endl;
		scriptOS << "set -e" << std::endl;
		scriptOS << std::endl;
		
		scriptOS << "lasttime=0" << std::endl;
		for (int i=0; i < step; i++) {
			std::string stepBase;
			{
				std::ostringstream oss;
				oss << dir << "/" << filenameBase << "-step";
				oss.width(8); oss.fill('0'); oss << i;
				stepBase = oss.str();
			}
			scriptOS << "currenttime=$(date +%s)" << std::endl;
			scriptOS << "if [ ${currenttime} -gt ${lasttime} ] ; then" << std::endl;
			scriptOS << "	echo Generating step " << i+1 << "/" << step << std::endl;
			scriptOS << "	lasttime=${currenttime}" << std::endl;
			scriptOS << "fi" << std::endl;
			scriptOS << "dot -Gfontname=Helvetica -Nfontname=Helvetica -Efontname=Helvetica -Tpdf " << stepBase << ".dot -o " << stepBase << ".pdf" << std::endl;
		}
		scriptOS << std::endl;
		
		scriptOS << "echo Joining into a single file" << std::endl;
		scriptOS << "pdfjoin -q --preamble '\\usepackage{hyperref} \\hypersetup{pdfpagelayout=SinglePage}' --outfile " << filenameBase << ".pdf";
		for (int i=0; i < step; i++) {
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
