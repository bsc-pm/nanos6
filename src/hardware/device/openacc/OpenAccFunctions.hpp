/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef OPENACC_FUNCTIONS_HPP
#define OPENACC_FUNCTIONS_HPP

#include <openacc.h>

#include "lowlevel/EnvironmentVariable.hpp"

// A helper class, providing static helper functions, specific to the device,
// to be used by DeviceInfo and other relevant classes as utilities.
// Unfortunately OpenACC does not -yet- have any error code returning mechanism;
// therefore these calls are inevitably left unchecked.
class OpenAccFunctions {
private:
	// TODO: handle types. As long as it works leave it like that
	static const acc_device_t _accDevType = acc_device_default;

public:
	static inline size_t getDeviceCount()
	{
		return (size_t)acc_get_num_devices(_accDevType);
	}

	static inline void setActiveDevice(int device)
	{
		acc_set_device_num(device, _accDevType);
	}

	static inline bool asyncFinished(int queue)
	{
		return (acc_async_test(queue) != 0);
	}

	static inline size_t getInitialQueueNum()
	{
		static EnvironmentVariable<size_t> initQueues("NANOS6_OPENACC_DEFAULT_QUEUES", 64);
		return initQueues;
	}

	static inline size_t getMaxQueues()
	{
		static EnvironmentVariable<size_t> maxQueues("NANOS6_OPENACC_MAX_QUEUES", 128);
		return maxQueues;
	}

};

#endif // OPENACC_FUNCTIONS_HPP

