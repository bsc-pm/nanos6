#ifndef NULL_HARDWARE_COUNTERS_HPP
#define NULL_HARDWARE_COUNTERS_HPP


class Task;

class HardwareCounters {

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
	
	static inline void startTaskMonitoring(Task *)
	{
	}
	
	static inline void stopTaskMonitoring(Task *)
	{
	}
	
	static inline void taskFinished(Task *)
	{
	}
	
	static inline void initializeThread()
	{
	}
	
	static inline void shutdownThread()
	{
	}
	
};

#endif // NULL_HARDWARE_COUNTERS_HPP
