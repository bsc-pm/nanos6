#ifndef TASKLOOP_MANAGER_HPP
#define TASKLOOP_MANAGER_HPP

#include <nanos6.h>

// Forward declaration
class Task;
class Taskloop;

class TaskloopManager {
public:
	static void handleTaskloop(Taskloop *runnableTaskloop, Taskloop *sourceTaskloop);
	static inline Taskloop* createRunnableTaskloop(Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds);
	static inline Taskloop* createPartitionTaskloop(Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds);
	static inline void completeTaskloopCreation(Taskloop *taskloop, Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds);
};

#endif // TASKLOOP_MANAGER_HPP
