/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_INFO_HPP
#define TASK_INFO_HPP

#include <atomic>
#include <map>
#include <string>

#include <nanos6/task-info-registration.h>

#include "lowlevel/SpinLock.hpp"
#include "tasks/TaskTypeData.hpp"


class TaskInfo {

private:

	struct TaskTypeId {
		std::string _taskLabel;
		std::string _taskDeclarationSource;

		inline TaskTypeId(
			const std::string &taskLabel,
			const std::string &taskDeclarationSource
		) :
			_taskLabel(taskLabel),
			_taskDeclarationSource(taskDeclarationSource)
		{
		}

		//! \brief Overloaded lesser-operator for task_type_map_t
		//! Duplicated TaskInfos have the same label or the same declaration
		//! source, thus we must overload the operator to act this way
		bool operator<(const TaskTypeId &other) const
		{
			assert(_taskLabel != "");
			assert(_taskDeclarationSource != "");
			assert(other._taskLabel != "");
			assert(other._taskDeclarationSource != "");

			// NOTE:
			// If either the label or the declaration source are the same,
			// both taskinfos are of the same tasktype, thus the comparison
			// must result in false. The map will compare both keys, and if the
			// following condition is met, the keys are treated as identical:
			// !(K1 < K2) && !(K2 < K1)
			//
			// -------------------------------------------
			// | Expected Comparison Results             |
			// -------------------------------------------
			// | Labels    | DeclarationSources | Result |
			// |-----------+--------------------+---------
			// | Equal     | Equal              | False  |
			// | Equal     | Different          | False  |
			// | Different | Equal              | False  |
			// | Different | Different          | True   |
			// -------------------------------------------

			bool sameLabel = (_taskLabel == other._taskLabel);
			bool sameSource = (_taskDeclarationSource == other._taskDeclarationSource);

			if (sameLabel || sameSource) {
				return false;
			} else if (!sameLabel) {
				return (_taskLabel < other._taskLabel);
			} else {
				return (_taskDeclarationSource < other._taskDeclarationSource);
			}
		}
	};

	typedef std::map<TaskTypeId, TaskTypeData> task_type_map_t;

	//! A map with task type data
	static task_type_map_t _taskTypes;

	//! SpinLock to register taskinfos
	static SpinLock _lock;

	//! Keep track of the number of unlabeled tasktypes
	static std::atomic<size_t> _numUnlabeledTaskTypes;

public:

	//! \brief Register the taskinfo of a type of task
	//!
	//! \param[in,out] taskInfo A pointer to the taskinfo
	static void registerTaskInfo(nanos6_task_info_t *taskInfo);

};

#endif // TASK_INFO_HPP
