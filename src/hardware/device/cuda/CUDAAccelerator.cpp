/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2024 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <array>

#include "CUDAAccelerator.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/MathSupport.hpp"
#include "system/BlockingAPI.hpp"

#include <DataAccessRegistration.hpp>
#include <DataAccessRegistrationImplementation.hpp>
#include <InstrumentWorkerThread.hpp>


ConfigVariable<bool> CUDAAccelerator::_pinnedPolling("devices.cuda.polling.pinned");
ConfigVariable<size_t> CUDAAccelerator::_usPollingPeriod("devices.cuda.polling.period_us");
ConfigVariable<bool> CUDAAccelerator::_prefetchDataDependencies("devices.cuda.prefetch");

thread_local Task *CUDAAccelerator::_currentTask;


void CUDAAccelerator::acceleratorServiceLoop()
{
	const size_t sleepTime = _usPollingPeriod.getValue();

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	while (!shouldStopService()) {
		bool activeDevice = false;
		do {
			// Launch as many ready device tasks as possible
			while (_streamPool.streamAvailable()) {
				Task *task = Scheduler::getReadyTask(_computePlace, currentThread);
				if (task == nullptr) {
					// The scheduler might have reported the thread as resting
					Instrument::workerProgressing();
					break;
				}

				runTask(task);
			}

			// Process events from Directory
			if (_directoryAgent != nullptr) {
				_directoryAgent->processEvents();
			}

			// Only set the active device if there have been tasks launched
			// Setting the device during e.g. bootstrap caused issues
			if (!_activeEvents.empty()) {
				if (!activeDevice) {
					activeDevice = true;
					setActiveDevice();
				}

				// Process the active events
				processCUDAEvents();
			}

			// Iterate while there are running tasks and pinned polling is enabled
		} while (_pinnedPolling && !_activeEvents.empty());

		// Sleep for a configured amount of microseconds
		BlockingAPI::waitForUs(sleepTime);
	}
}

// For each CUDA device task a CUDA stream is required for the asynchronous
// launch; To ensure kernel completion a CUDA event is 'recorded' on the stream
// right after the kernel is queued. Then when a cudaEventQuery call returns
// succefully, we can be sure that the kernel execution (and hence the task)
// has finished.

// Get a new CUDA event and queue it in the stream the task has launched
void CUDAAccelerator::postRunTask(Task *task)
{
	nanos6_cuda_device_environment_t &env = task->getDeviceEnvironment().cuda;
	CUDAFunctions::recordEvent(env.event, env.stream);
	_activeEvents.push_back({env.event, task});
}

void CUDAAccelerator::preRunTask(Task *task)
{
	// Set the thread_local static var to be used by nanos6_get_current_cuda_stream()
	CUDAAccelerator::_currentTask = task;

	// Nothing else to do if prefetch is disabled
	if (!_prefetchDataDependencies)
		return;

	// Disable prefetch when using memory directory
	if (Directory::isEnabled())
		return;

	// Prefetch available memory locations to the GPU
	nanos6_cuda_device_environment_t &env = task->getDeviceEnvironment().cuda;

	DataAccessRegistration::processAllDataAccesses(task,
		[&](const DataAccess *access) -> bool {
			if (access->getType() != REDUCTION_ACCESS_TYPE && !access->isWeak()) {
				CUDAFunctions::prefetchMemory(
					access->getAccessRegion().getStartAddress(),
					access->getAccessRegion().getSize(),
					_deviceHandler, env.stream,
					access->getType() == READ_ACCESS_TYPE);
			}
			return true;
		});
}

// Query the events issued to detect task completion
void CUDAAccelerator::processCUDAEvents()
{
	_preallocatedEvents.clear();
	std::swap(_preallocatedEvents, _activeEvents);

	for (CUDAEvent &ev : _preallocatedEvents) {
		if (CUDAFunctions::isEventFinished(ev.event)) {
			finishTask(ev.task);
		} else {
			_activeEvents.push_back(ev);
		}
	}
}


void CUDAAccelerator::callTaskBody(Task *task, nanos6_address_translation_entry_t *translationTable)
{
	nanos6_task_info_t *taskInfo = task->getTaskInfo();
	assert(taskInfo != nullptr);

	nanos6_cuda_device_environment_t &env = task->getDeviceEnvironment().cuda;

	void *args = task->getArgsBlock();
	nanos6_device_info_t &deviceInfo = *((nanos6_device_info_t *)args);

	// The ndrange clause in device tasks is used to define the work hierarchy
	// used to execute the kernel in the device. The work hierarchy can be 1D,
	// 2D, or 3D. The first parameter of the clause is the number of dimensions,
	// the next are the global sizes of elements in each dimension, and then,
	// the last are the local sizes of elements in each dimenion. The global
	// sizes indicate how many threads will be spawn in total. The local sizes
	// indicate the number of threads per CUDA block
	//
	// NOTE: The global sizes do not correspond to the CUDA grid sizes
	//
	// TODO: If the parameters provided by the user are invalid for our CUDA
	// capabilities, we could perform some math to express the same working
	// units in a supported way. Right now we expect the user to provide valid
	// parameters

	// The device task may need the body to run on the host
	const char *kernelName = taskInfo->implementations[0].device_function_name;
	if (deviceInfo.sizes[0] == -1 && deviceInfo.sizes[3] == -1) {
		return task->body(translationTable);
	}

	// Retrieve the local sizes (CUDA block sizes)
	size_t blockDim1 = std::max((int64_t) deviceInfo.sizes[3], (int64_t) 1);
	size_t blockDim2 = std::max((int64_t) deviceInfo.sizes[4], (int64_t) 1);
	size_t blockDim3 = std::max((int64_t) deviceInfo.sizes[5], (int64_t) 1);

	// Retrieve the global sizes (not the CUDA grid sizes)
	size_t globalDim1 = std::max((int64_t) deviceInfo.sizes[0], (int64_t) 1);
	size_t globalDim2 = std::max((int64_t) deviceInfo.sizes[1], (int64_t) 1);
	size_t globalDim3 = std::max((int64_t) deviceInfo.sizes[2], (int64_t) 1);

	// Compute the CUDA grid sizes using the global and local dimensions
	size_t gridDim1 = MathSupport::ceil(globalDim1, blockDim1);
	size_t gridDim2 = MathSupport::ceil(globalDim2, blockDim2);
	size_t gridDim3 = MathSupport::ceil(globalDim3, blockDim3);

	std::array<void *, MAX_STACK_ARGS> stackParams;
	void **params = &stackParams[0];
	int numArgs = taskInfo->num_args;

	if (numArgs > MAX_STACK_ARGS)
		params = (void **) MemoryAllocator::alloc(numArgs * sizeof(void *));

	for (int i = 0; i < numArgs; i++)
		params[i] = (void *) ((char *) args + taskInfo->offset_table[i]);

	if (translationTable) {
		for (int i = 0; i < taskInfo->num_symbols; ++i) {
			int arg = taskInfo->arg_idx_table[i];
			// Translate corresponding parameter
			if (arg >= 0 && translationTable[i].device_address != 0) {
				// The params[arg] is a void pointer which actually points to
				// the location of the argument. We want to translate is the
				// argument itself, so we need to dereference it
				uintptr_t *argument = (uintptr_t *) params[arg];
				*argument = *argument - translationTable[i].local_address
					+ translationTable[i].device_address;
			}
		}
	}

	// Launch the kernel
	CUDAFunctions::launchKernel(kernelName, params,
		gridDim1, gridDim2, gridDim3,
		blockDim1, blockDim2, blockDim3,
		deviceInfo.shm_size, env.stream);

	// Un-translate the arguments, in case this task needs to be realunched at some point
	if (translationTable) {
		for (int i = 0; i < taskInfo->num_symbols; ++i) {
			int arg = taskInfo->arg_idx_table[i];
			if (arg >= 0 && translationTable[i].device_address != 0) {
				uintptr_t *argument = (uintptr_t *)params[arg];
				*argument = *argument - translationTable[i].device_address
					+ translationTable[i].local_address;
			}
		}
	}

	// Free the arguments if necessary
	if (numArgs > MAX_STACK_ARGS)
		MemoryAllocator::free((void *)params, numArgs * sizeof(void *));
}
