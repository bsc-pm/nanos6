/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_COMPUTE_PLACE_HPP
#define DEVICE_COMPUTE_PLACE_HPP

#include <hardware/places/ComputePlace.hpp>

class DeviceMemoryPlace;
class Task;
class DeviceFunctionsInterface;

class DeviceComputePlace: public ComputePlace {
	nanos6_device_t _type;
	int _subType;
	void *_deviceHandler;
	int _maxRunningTasks;
	std::atomic<int> _runningTasks;
	std::string _strPollingService;
	DeviceFunctionsInterface *_functions;
	DeviceMemoryPlace *_memoryPlace;
	
	static inline int deviceMaxRunningTask(nanos6_device_t dev);
public:
	
	DeviceComputePlace(DeviceMemoryPlace *memoryPlace, nanos6_device_t type, int subType,
			int index, DeviceFunctionsInterface* functions, void *deviceHandler);
	
	~DeviceComputePlace();
	
	DeviceMemoryPlace *getMemoryPlace();
	
	/*! \brief Execute the body of a task */
	void runTask(Task *task);
	
	bool isReady();
	
	int getNumUnFinishedTasks();
	
	int getMaxRunningTasks();
	
	int getRunningTasks();
	
	bool canRunTask();
	
	void disposeTask();
	
	static int pollingFinishTasks(DeviceFunctionsInterface *functions);
	
	static int pollingRun(DeviceComputePlace *computePlace);
	
	void deactivatePollingService();
	
	void activatePollingService();
	
	int getSubType();
	
	int getType();
	
};

#endif //DEVICE_COMPUTE_PLACE_HPP
