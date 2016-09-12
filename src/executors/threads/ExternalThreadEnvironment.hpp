#ifndef EXTERNAL_THREAD_ENVIRONMENT_HPP
#define EXTERNAL_THREAD_ENVIRONMENT_HPP


#include "CPUDependencyData.hpp"
#include "EssentialThreadEnvironment.hpp"


class ExternalThreadEnvironment : public EssentialThreadEnvironment {
private:
	CPUDependencyData _cpuDependencyData;
	
	static __thread ExternalThreadEnvironment *_taskWrapperEvironment;
	
	
public:
	ExternalThreadEnvironment();
	virtual ~ExternalThreadEnvironment();
	
	//! \brief This method should only be called by non-worker threads to allocate an environment in which to create tasks
	static ExternalThreadEnvironment *getTaskWrapperEnvironment()
	{
		if (_taskWrapperEvironment == nullptr) {
			_taskWrapperEvironment = new ExternalThreadEnvironment();
		}
		
		return _taskWrapperEvironment;
	}
	
	CPUDependencyData &getDependencyData()
	{
		return _cpuDependencyData;
	}
	
};


#endif // EXTERNAL_THREAD_ENVIRONMENT_HPP
