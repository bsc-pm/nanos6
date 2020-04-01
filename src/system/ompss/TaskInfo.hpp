/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_INFO_HPP
#define TASK_INFO_HPP

#include <atomic>
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
		nanos6_task_info_t *duplicateTaskInfo
	) {
		// No duplicates for now
		return false;
	}

public:

	//! \brief Register the taskinfo of a type of task
	//!
	//! \param[in,out] taskInfo A pointer to the taskinfo
	static void registerTaskInfo(nanos6_task_info_t *taskInfo);

};

#endif // TASK_INFO_HPP
