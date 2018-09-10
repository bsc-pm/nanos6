/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_HELPER_HPP
#define CUDA_HELPER_HPP

#include <string>

#include "lowlevel/SpinLock.hpp" 

#include "hardware/cuda/CUDADevice.hpp"

class Task;

class CUDAPollingService {

private:
	std::string _serviceName;
	CUDADevice *_device;
	SpinLock _lock;
	
	void finishTask(Task *task);
	void launchTask(Task *task);	
public:	
	CUDAPollingService(CUDADevice *device);
	
	CUDAPollingService(CUDAPollingService const &) = delete;
	CUDAPollingService operator=(CUDAPollingService const &) = delete;
	
	~CUDAPollingService();
	
	//! \brief Register the polling service 
	void start();
	
	//! \brief Unregister the polling service 
	void stop();
	
	//! \brief Main function of the helper 	
	void run();
	
	//! \brief Static function used by the PollingAPI 
	static bool runHelper(void *helper_ptr);
};


#endif // CUDA_HELPER_HPP

