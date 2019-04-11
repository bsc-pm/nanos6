/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>

#include "InstrumentHardwareCounters.hpp"
#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentStats.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "system/RuntimeInfo.hpp"

#include <executors/threads/CPUManager.hpp>
#include <executors/threads/ThreadManager.hpp>


namespace Instrument {
	namespace Stats {
		static void emitTaskInfo(std::ofstream &output, std::string const &name, TaskInfo &taskInfo)
		{
			TaskTimes meanTimes = taskInfo._times / taskInfo._numInstances;
			
			double meanLifetime = meanTimes.getTotal();
			
			output << "STATS\t" << name << " instances\t"
				<< taskInfo._numInstances << std::endl;
			output << "STATS\t" << name << " mean instantiation time\t"
				<< meanTimes._instantiationTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._instantiationTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean pending time\t"
				<< meanTimes._pendingTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._pendingTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean ready time\t"
				<< meanTimes._readyTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._readyTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean execution time\t"
				<< meanTimes._executionTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._executionTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean blocked time\t"
				<< meanTimes._blockedTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._blockedTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean zombie time\t"
				<< meanTimes._zombieTime << "\t" << Timer::getUnits()
				<< "\t" << 100.0 * (double) meanTimes._zombieTime / meanLifetime << "\t%" << std::endl;
			output << "STATS\t" << name << " mean lifetime\t"
				<< meanTimes.getTotal() << "\t" << Timer::getUnits() << std::endl;
			
			for (InstrumentHardwareCounters::counter_value_t const &counterValue : taskInfo._hardwareCounters[0]) {
				output << "STATS\t" << name << " " << counterValue._name << "\t";
				
				if (counterValue._isInteger) {
					output << counterValue._integerValue;
				} else {
					output << counterValue._floatValue;
				}
				
				if (!counterValue._units.empty()) {
					output << "\t" << counterValue._units;
				}
				
				output << std::endl;
			}
			
		}
	}
	
	
	using namespace Stats;
	
	
	void initialize()
	{
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "stats");
		InstrumentHardwareCounters::initialize();
		
		_phasesSpinLock.writeLock();
		_phaseTimes.emplace_back(true);
		_phasesSpinLock.writeUnlock();
	}
	
	
	void shutdown()
	{
		_totalTime.stop();
		double totalTime = _totalTime;
		
		InstrumentHardwareCounters::shutdown();
		
		ThreadInfo accumulatedThreadInfo(false);
		int numThreads = 0;
		for (auto &threadInfo : _threadInfoList) {
			threadInfo->getCurrentPhaseRef().stoppedAt(_totalTime);
			
			accumulatedThreadInfo += *threadInfo;
			numThreads++;
		}
		
		PhaseInfo accumulatedPhaseInfo(false);
		for (auto &phaseInfo : accumulatedThreadInfo._phaseInfo) {
			accumulatedPhaseInfo += phaseInfo;
		}
		
		double totalThreadTime = (double) accumulatedPhaseInfo._blockedTime + (double) accumulatedPhaseInfo._runningTime;
		double totalRunningTime = accumulatedPhaseInfo._runningTime;
		
		double averageThreadTime = totalThreadTime / (double) numThreads;
		
		TaskInfo accumulatedTaskInfo;
		for (auto &taskInfoEntry : accumulatedPhaseInfo._perTask) {
			accumulatedTaskInfo += taskInfoEntry.second;
		}
		
		EnvironmentVariable<std::string> _outputFilename("NANOS6_STATS_FILE", "/dev/stderr");
		std::ofstream output(_outputFilename);
		
		output << "STATS\t" << "Total CPUs\t" << CPUManager::getTotalCPUs() << std::endl;
		output << "STATS\t" << "Total time\t" << totalTime << "\tns" << std::endl;
		output << "STATS\t" << "Total threads\t" << numThreads << std::endl;
		output << "STATS\t" << "Mean threads per CPU\t" << ((double) numThreads) / (double) CPUManager::getTotalCPUs() << std::endl;
		output << "STATS\t" << "Mean tasks per thread\t" << ((double) accumulatedTaskInfo._numInstances) / (double) numThreads << std::endl;
		output << std::endl;
		output << "STATS\t" << "Mean thread lifetime\t" << 100.0 * averageThreadTime / totalTime << "\t%" << std::endl;
		output << "STATS\t" << "Mean thread running time\t" << 100.0 * totalRunningTime / totalThreadTime << "\t%" << std::endl;
		output << "STATS\t" << "Mean effective parallelism\t" << (double) accumulatedTaskInfo._times._executionTime / (double) totalTime << std::endl;
		
		if (accumulatedTaskInfo._numInstances > 0) {
			output << std::endl;
			emitTaskInfo(output, "All Tasks", accumulatedTaskInfo);
		}
		
		
		for (auto &taskInfoEntry : accumulatedPhaseInfo._perTask) {
			nanos6_task_info_t const *userSideTaskInfo = taskInfoEntry.first;
			
			assert(userSideTaskInfo != 0);
			std::string name;
			if ((userSideTaskInfo->implementations[0].task_label != nullptr) && (userSideTaskInfo->implementations[0].task_label[0] != '\0')) {
				name = userSideTaskInfo->implementations[0].task_label;
			} else if (userSideTaskInfo->implementations[0].declaration_source != 0) {
				name = userSideTaskInfo->implementations[0].declaration_source;
			} else {
				name = "Unknown task";
			}
			
			output << std::endl;
			emitTaskInfo(output, name, taskInfoEntry.second);
		}
		
		{
			int phase = 0;
			for (auto &phaseInfo : accumulatedThreadInfo._phaseInfo) {
				TaskInfo currentPhaseAccumulatedTaskInfo;
				
				for (auto &taskInfoEntry : phaseInfo._perTask) {
					nanos6_task_info_t const *userSideTaskInfo = taskInfoEntry.first;
					
					assert(userSideTaskInfo != 0);
					std::string name;
					if ((userSideTaskInfo->implementations[0].task_label != nullptr) && (userSideTaskInfo->implementations[0].task_label[0] != '\0')) {
						name = userSideTaskInfo->implementations[0].task_label;
					} else if (userSideTaskInfo->implementations[0].declaration_source != 0) {
						name = userSideTaskInfo->implementations[0].declaration_source;
					} else {
						name = "Unknown task";
					}
					
					if (name == "main") {
						// Main ends up in the last phase despite the fact that it contributes to all of them
						continue;
					}
					
					std::ostringstream oss;
					oss << "Phase " << (phase+1) << " " << name;
					
					output << std::endl;
					emitTaskInfo(output, oss.str(), taskInfoEntry.second);
					
					currentPhaseAccumulatedTaskInfo += taskInfoEntry.second;
				}
				
				if (currentPhaseAccumulatedTaskInfo._numInstances > 0) {
					std::ostringstream oss;
					oss << "Phase " << (phase+1);
					
					output << std::endl;
					emitTaskInfo(output, oss.str(), currentPhaseAccumulatedTaskInfo);
					output << "STATS\t" << "Phase " << (phase+1) << " effective parallelism\t"
						<< (double) currentPhaseAccumulatedTaskInfo._times._executionTime / (double) _phaseTimes[phase] << std::endl;
				}
				
				phase++;
			}
		}
		
		output.close();
	}
}


