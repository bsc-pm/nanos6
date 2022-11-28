/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_INFO_DATA_HPP
#define TASK_INFO_DATA_HPP

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <map>

#include <nanos6/task-info-registration.h>

#include "lowlevel/SpinLock.hpp"

#include <InstrumentTasktypeData.hpp>


class TaskInfoManager;
class TasktypeStatistics;

class TaskInfoData {
	//! Task type label
	std::string _taskTypeLabel;

	//! Task declaration source
	std::string _taskDeclarationSource;

	//! Instrumentation identifier for this task info
	Instrument::TasktypeInstrument _instrumentId;

	//! Monitoring-related statistics per taskType
	TasktypeStatistics *_taskTypeStatistics;

	friend class TaskInfoManager;

public:
	inline TaskInfoData() :
		_taskTypeLabel(),
		_taskDeclarationSource(),
		_instrumentId(),
		_taskTypeStatistics(nullptr)
	{
	}

	inline const std::string &getTaskTypeLabel() const
	{
		return _taskTypeLabel;
	}

	inline const std::string &getTaskDeclarationSource() const
	{
		return _taskDeclarationSource;
	}

	inline Instrument::TasktypeInstrument &getInstrumentationId()
	{
		return _instrumentId;
	}
	
	inline void setTasktypeStatistics(TasktypeStatistics *taskTypeStatistics)
	{
		_taskTypeStatistics = taskTypeStatistics;
	}

	inline TasktypeStatistics *getTasktypeStatistics() const
	{
		return _taskTypeStatistics;
	}
};

class TaskInfoManager {
	//! A map with task info data useful for filtering duplicated task infos
	typedef std::map<nanos6_task_info_t *, TaskInfoData> task_info_map_t;
	static task_info_map_t _taskInfos;

	//! SpinLock to register and traverse task infos
	static SpinLock _lock;

	//! Check whether any device kernel must be loaded
	static void checkDeviceTaskInfo(const nanos6_task_info_t *taskInfo);

public:
	//! \brief Register a new task info
	//!
	//! \param[in,out] taskInfo A pointer to a task info
	static inline TaskInfoData &registerTaskInfo(nanos6_task_info_t *taskInfo) {
		assert(taskInfo != nullptr);
		assert(taskInfo->implementations != nullptr);
		assert(taskInfo->implementations[0].declaration_source != nullptr);

		// Global number of unlabeled task infos
		static size_t unlabeledTaskInfos = 0;

		// Check whether any device kernel must be loaded
		checkDeviceTaskInfo(taskInfo);

		std::lock_guard<SpinLock> lock(_lock);

		auto result = _taskInfos.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(taskInfo),
			std::forward_as_tuple(/* empty */)
		);

		TaskInfoData &data = result.first->second;

		// Stop since it's not a new task info
		if (!result.second)
			return data;

		// Setup task type label and declaration source
		if (taskInfo->implementations[0].task_type_label) {
			data._taskTypeLabel = std::string(taskInfo->implementations[0].task_type_label);
		} else {
			data._taskTypeLabel = "Unlabeled" + std::to_string(unlabeledTaskInfos++);
		}
		data._taskDeclarationSource = std::string(taskInfo->implementations[0].declaration_source);

		// Setup the reference to the data in the task info
		taskInfo->task_type_data = &data;

		return data;
	}

	//! \brief Traverse all task infos and apply a certain function for each of them
	//!
	//! \param[in] functionToApply The function to apply to each task info
	template <typename F>
	static inline void processAllTaskInfos(F functionToApply)
	{
		std::lock_guard<SpinLock> lock(_lock);

		for (auto &taskInfo : _taskInfos) {
			functionToApply(taskInfo.first, taskInfo.second);
		}
	}
};

#endif // TASK_INFO_DATA_HPP
