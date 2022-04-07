/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <string>

#include "TaskInfo.hpp"


#ifdef USE_CUDA
#include "hardware/device/cuda/CUDAFunctions.hpp"
#endif

TaskInfo::task_type_map_t TaskInfo::_tasktypes;
SpinLock TaskInfo::_lock;
std::atomic<size_t> TaskInfo::_numUnlabeledTasktypes(0);


bool TaskInfo::registerTaskInfo(nanos6_task_info_t *taskInfo)
{
	assert(taskInfo != nullptr);
	assert(taskInfo->implementations != nullptr);
	assert(taskInfo->implementations->declaration_source != nullptr);

#ifdef USE_CUDA
	// Check that all cuda tasks have a valid implementation on the GPU
	if (CUDAFunctions::initialize()) {
		if (taskInfo->implementations[0].device_type_id == nanos6_cuda_device && taskInfo->implementations[0].device_function_name != nullptr) {
			for (size_t gpu = 0; gpu < CUDAFunctions::getDeviceCount(); ++gpu) {
				CUDAFunctions::loadFunction(taskInfo->implementations[0].device_function_name);
			}
		}
	}
#endif

	std::string label;
	if (taskInfo->implementations->task_type_label != nullptr) {
		label = std::string(taskInfo->implementations->task_type_label);
	} else {
		// Avoid comparing empty strings and identify them separately
		size_t unlabeledId = _numUnlabeledTasktypes++;
		label = "Unlabeled" + std::to_string(unlabeledId);
	}

	std::string declarationSource(taskInfo->implementations->declaration_source);

	// NOTE: We try to emplace the new TaskInfo in the map:
	// 1) If the element is emplaced, it's a new type of task and a new
	// TasktypeData has been created
	// 2) If the key already existed, it's a duplicated type of task, and the
	// iterator points to the original copy
	std::pair<task_type_map_t::iterator, bool> emplacedElement;

	_lock.lock();

	emplacedElement = _tasktypes.emplace(
		std::piecewise_construct,
		std::forward_as_tuple(label, declarationSource),
		std::forward_as_tuple());

	_lock.unlock();

	// Save a reference of this task type in the task info
	task_type_map_t::iterator it = emplacedElement.first;
	taskInfo->task_type_data = &(it->second);

	// true if new element, false if already existed
	return emplacedElement.second;
}
