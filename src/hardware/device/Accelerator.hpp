/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef ACCELERATOR_HPP
#define ACCELERATOR_HPP

#include <hardware/places/ComputePlace.hpp>
#include <hardware/places/MemoryPlace.hpp>
#include <tasks/Task.hpp>


// The Accelerator class should be used *per physical device*,
// to denote a separate address space. Accelerators may have
// sub-devices, if applicable, for instance in FPGAs, that
// virtual partitions of the FPGA may share the address space.
// With GPUs, each physical GPU gets its Accelerator instance.

class Accelerator {
protected:

	// Used also to denote the device number
	int _deviceHandler;
	nanos6_device_t _deviceType;
	MemoryPlace *_memoryPlace;
	ComputePlace *_computePlace;

	Accelerator(int handler, nanos6_device_t type) :
		_deviceHandler(handler),
		_deviceType(type)
	{
		_memoryPlace = new MemoryPlace(_deviceHandler, _deviceType);
		_computePlace = new ComputePlace(_deviceHandler, _deviceType);
		_computePlace->addMemoryPlace(_memoryPlace);
	}

	// Set the current instance as the selected/active device for subsequent operations
	virtual void setActiveDevice() = 0;

	// Each Accelerator needs to implement a pollingService(), to be registered and handle task launch/completion.
	// The polling service needs to be declared as a *public* method:
	// static int pollingService(void *data);

	virtual void registerPolling() = 0;

	virtual void unregisterPolling() = 0;

	virtual void acceleratorServiceLoop() = 0;

	// Each device may use these methods to prepare or conclude task launch if needed
	virtual void preRunTask(Task *)
	{
	}

	virtual void postRunTask(Task *)
	{
	}

	// The main device task launch method; It will call pre- & postRunTask
	virtual void runTask(Task *task);

	// Device specific operations after task completion may go here (e.g. free environment)
	virtual void finishTaskCleanup(Task *)
	{
	}

	// Generate the appropriate device_env pointer Mercurium uses for device tasks
	virtual void generateDeviceEvironment(Task *)
	{
	}

	virtual void finishTask(Task *task);

public:
	virtual ~Accelerator()
	{
		delete _computePlace;
		delete _memoryPlace;
	}

	MemoryPlace *getMemoryPlace()
	{
		return _memoryPlace;
	}

	ComputePlace *getComputePlace()
	{
		return _computePlace;
	}

	nanos6_device_t getDeviceType() const
	{
		return _deviceType;
	}

	int getDeviceHandler() const
	{
		return _deviceHandler;
	}

	// We use FIFO queues that we launch the tasks on. These are used to check
	// for task completion or to enqueue related operations (in the case of
	// the Cache-Directory).  The actual object type is device-specific: In
	// CUDA, this will be a cudaStream_t, in OpenACC an asynchronous queue,
	// in FPGA a custom FPGA stream, etc.

	// This call requests an available FIFO from the runtime and returns a pointer to it.
	virtual void *getAsyncHandle() = 0;

	// Return the FIFO for re-use after task has finished.
	virtual void releaseAsyncHandle(void *asyncHandle) = 0;

};

#endif // ACCELERATOR_HPP

