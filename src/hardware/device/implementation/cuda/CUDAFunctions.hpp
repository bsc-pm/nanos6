/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_CUDAFUNCTIONS_HPP
#define CUDA_CUDAFUNCTIONS_HPP

#include <cassert>
#include <cstdlib>
#include <queue>

#include <cuda_runtime_api.h>

#include <nanos6/cuda_device.h>

#include "CUDAContext.hpp"
#include "hardware/places/DeviceComputePlace.hpp"
#include "hardware/places/DeviceMemoryPlace.hpp"
#include "hardware/device/DeviceFunctionsInterface.hpp"
#include "lowlevel/cuda/CUDAErrorHandler.hpp"
#include "lowlevel/SpinLock.hpp"
#include "memory/vmm/VirtualMemoryArea.hpp"
#include "tasks/Task.hpp"

class CUDAFunctions: public DeviceFunctionsInterface {

private:
	SpinLock _depsLock;
	const nanos6_device_t device_type = nanos6_cuda_device;
	bool _correctlyInitialized;

public:
	std::vector<std::pair<void *, CUDA_DEVICE_DEP *>> _cudaDeps;

	CUDAFunctions()
	{
		_devices.push_back(new Device(nanos6_cuda_device, 0));

		int deviceCount;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if (err != cudaSuccess) {
			_correctlyInitialized = false;
			if (err != cudaErrorNoDevice) {
				CUDAErrorHandler::warnIf(true, "Nanos6 was compiled with CUDA support but the driver returned: ",
					cudaGetErrorString(err),"\nRunning CUDA tasks is disabled");
			}
			return;
		}

		assert(deviceCount > 0);
		_cudaDeps.resize(deviceCount);

		for (int i = 0; i < deviceCount; ++i) {
			cudaSetDevice(i);

			DeviceComputePlace *cp = new DeviceComputePlace(new DeviceMemoryPlace(i, nanos6_cuda_device),
					nanos6_device_t::nanos6_cuda_device, 0, i, this, nullptr);
			_devices[0]->addComputePlace(cp);

			_cudaDeps[i].first = (void *) cp;
			_cudaDeps[i].second = new CUDA_DEVICE_DEP();
		}
		_correctlyInitialized  = true;
	}

	~CUDAFunctions()
	{
		for (size_t i = 0; i < _cudaDeps.size(); i++) {
			DeviceComputePlace *dcp = (DeviceComputePlace *) _cudaDeps[i].first;
			DeviceMemoryPlace *dmp = dcp->getMemoryPlace();
			delete dmp;
			delete _cudaDeps[i].second;
		}
	}

	void shutdown()
	{
		if (_correctlyInitialized) {
			nanos6_unregister_polling_service("taskFinisher",
					(nanos6_polling_service_t) DeviceComputePlace::pollingFinishTasks, this);
		}
	}

	size_t getComputePlaceCount()
	{
		return _cudaDeps.size();
	}

	ComputePlace *getComputePlace(int index)
	{
		return (ComputePlace *) _cudaDeps[index].first;
	}

	CUDA_DEVICE_DEP *getDeps(void *ptr)
	{
		for (auto p : _cudaDeps) {
			if (p.first == ptr)
				return p.second;
		}
		return nullptr;
	}

	int malloc(void **ptr, size_t size)
	{
		cudaMalloc(ptr, size);
		if (ptr != nullptr)
			return Success;
		else
			return ErrorMemoryAllocation;
	}

	void free(void *ptr)
	{
		cudaFree(ptr);
	}

	int memcpy(void *dst, void *src, size_t size, deviceMemcpy type)
	{
		return (cudaMemcpy(dst, src, size, (cudaMemcpyKind) type) == cudaSuccess);
	}

	int setDevice(int device)
	{
		return (cudaSetDevice(device) == cudaSuccess);
	}

	const char *getName()
	{
		static std::string *str = new std::string("CUDA");
		return str->c_str();
	}

	void runTask(Task *task, ComputePlace *cp)
	{
		assert(task != nullptr);
		assert(cp != nullptr);

		((DeviceComputePlace *) cp)->runTask(task);
	}

	nanos6_device_t getType()
	{
		return device_type;
	}

	void *generateDeviceExtra(Task *task, void *)
	{
		nanos6_cuda_device_environment_t *env =
				(nanos6_cuda_device_environment_t *) ::malloc(
						sizeof(nanos6_cuda_device_environment_t));

		CUDAStream *deviceDataStream  = getDeps(task->getComputePlace())->_pool.getStream();
		task->setDeviceData((void *) deviceDataStream);
		env->stream = deviceDataStream->getStream();

		return (void *) env;
	}

	void postBodyDevice(Task *task, void *)
	{
		auto taskcp = task->getComputePlace();
		auto deps = getDeps(taskcp);
		CUDAEvent *event = deps->_eventPool.getEvent();

		event->setTask(task);
		event->record();
		std::lock_guard<SpinLock> guard(_depsLock);
		getDeps(task->getComputePlace())->_activeEvents.push_back(event);
	}

	void bodyDevice(Task *, void *)
	{

	}
	void getFinishedTasks(std::vector<Task *>& finishedTasks)
	{
		for (unsigned int i = 0; i < _cudaDeps.size(); ++i) {
			std::vector<CUDAEvent *> *_activeEvents = &_cudaDeps[i].second->_activeEvents;
			if (_activeEvents != nullptr) {

				std::lock_guard<SpinLock> guard(_depsLock);
				auto it = _activeEvents->begin();
				while (it != _activeEvents->end()) {

					if ((*it)->finished()) {
						CUDAEvent *evt = *it;

						_cudaDeps[i].second->_eventPool.returnEvent(evt);
						finishedTasks.push_back(evt->getTask());
						Task *task = evt->getTask();
						if (task != nullptr) {
							DeviceComputePlace *devComputePlace =
									(DeviceComputePlace *) (task->getComputePlace());

							devComputePlace->disposeTask();
							CUDA_DEVICE_DEP* deps = getDeps(task->getComputePlace());
							deps->_pool.returnStream((CUDAStream *) task->getDeviceData());
						}

						it = _activeEvents->erase(it);
					}
					else {
						++it;
					}
				}

			}
		}

	}

	void unifiedAsyncPrefetch(void *pHost, size_t size, int dstDevice)
	{
		setDevice(dstDevice);

		void *dev;
		cudaHostGetDevicePointer(&dev, pHost, 0);
		memcpy(pHost, dev, size, HOST_TO_DEVICE);

	}

	void *unifiedGetDevicePointer(void *pHost)
	{
		void *dev;
		cudaHostGetDevicePointer(&dev, pHost, 0);
		return dev;
	}

	void unifiedMemRegister(void *pHost, size_t size)
	{
		cudaHostRegister(pHost, size, cudaHostRegisterDefault);
	}

	void unifiedMemUnregister(void *pHost)
	{
		cudaHostUnregister(pHost);
	}

	bool initialize()
	{
		if (_correctlyInitialized) {
			nanos6_register_polling_service("taskFinisher",
				(nanos6_polling_service_t) DeviceComputePlace::pollingFinishTasks, this);

			for (Device *device : _devices) {
				for (int i = 0; i < device->getNumDevices(); ++i) {
					device->getComputePlace(i)->activatePollingService();
				}
			}
		}

		return _correctlyInitialized;
	}

	bool getInitStatus()
	{
		return _correctlyInitialized;
	}

	void getDevices(std::vector<Device *> &dev)
	{
		dev = _devices;
	}

};

#endif // CUDA_CUDAFUNCTIONS_HPP
