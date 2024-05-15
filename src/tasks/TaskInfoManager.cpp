/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#include "TaskInfoManager.hpp"

#ifdef USE_CUDA
#include "hardware/device/cuda/CUDAFunctions.hpp"
#endif

TaskInfoManager::task_info_map_t TaskInfoManager::_taskInfos;
SpinLock TaskInfoManager::_lock;

void TaskInfoManager::checkDeviceTaskInfo(__attribute__((unused)) const nanos6_task_info_t *taskInfo)
{
	// TODO: This should be hidden inside the CUDA module
#ifdef USE_CUDA
	// Check whether is a CUDA task
	if (taskInfo->implementations[0].device_type_id != nanos6_cuda_device)
		return;

	// There are CUDA tasks, initialize the CUDA module
	if (!CUDAFunctions::initialize())
		return;
#endif
}
