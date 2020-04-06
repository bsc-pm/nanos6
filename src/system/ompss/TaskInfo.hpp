/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_INFO_HPP
#define TASK_INFO_HPP

#include <atomic>
#include <cstring>
#include <vector>

#include <nanos6/task-info-registration.h>

#include "lowlevel/SpinLock.hpp"


class TaskInfo {

private:

	//! A vector to register all the taskinfos
	static std::vector<nanos6_task_info_t *> _taskInfos;

	//! The next taskinfo identifier
	static std::atomic<short> _nextId;

private:

	//! \brief Check whether a duplicate of a TaskInfo exists
	//!
	//! \param[in] taskInfo The TaskInfo to check for
	//! \param[out] duplicateTaskInfo If there was a duplicate, it is copied
	//! in this parameter
	static inline bool isDuplicated(
		nanos6_task_info_t *taskInfo,
		nanos6_task_info_t *&duplicateTaskInfo
	) {
		assert(taskInfo != nullptr);
		assert(taskInfo->implementations != nullptr);

		for (short i = 0; i < _nextId.load(); ++i) {
			assert(_taskInfos[i] != nullptr);
			assert(_taskInfos[i]->implementations != nullptr);

			// If the declaration sources are the same, this is a duplicated taskinfo
			char const *declarationSource = _taskInfos[i]->implementations->declaration_source;
			if (strcmp(declarationSource, taskInfo->implementations->declaration_source) == 0) {
				duplicateTaskInfo = _taskInfos[i];
				return true;
			}

			// If users specify identical task labels, we consider they are the same tasktype
			char const *label = _taskInfos[i]->implementations->task_label;
			if (strcmp(label, taskInfo->implementations->task_label) == 0) {
				duplicateTaskInfo = _taskInfos[i];
				return true;
			}
		}

		return false;
	}

public:

	//! \brief Register the taskinfo of a type of task
	//!
	//! \param[in,out] taskInfo A pointer to the taskinfo
	static void registerTaskInfo(nanos6_task_info_t *taskInfo);

};

#endif // TASK_INFO_HPP
