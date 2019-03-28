#ifndef NULL_MONITORING_HPP
#define NULL_MONITORING_HPP

#include "tasks/Task.hpp"


class Monitoring {

public:
	
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline bool isEnabled()
	{
		return false;
	}
	
	static inline void taskCreated(Task *)
	{
	}
	
	static inline void taskChangedStatus(Task *, monitoring_task_status_t)
	{
	}
	
	static inline void taskFinished(Task *)
	{
	}
	
};

#endif // NULL_MONITORING_HPP
