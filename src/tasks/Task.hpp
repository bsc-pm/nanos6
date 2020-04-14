/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HPP
#define TASK_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <set>
#include <string>

#include <nanos6.h>

#include "hardware-counters/TaskHardwareCounters.hpp"
#include "lowlevel/SpinLock.hpp"

#include <ClusterTaskContext.hpp>
#include <ExecutionWorkflow.hpp>
#include <InstrumentTaskId.hpp>
#include <TaskDataAccesses.hpp>
#include <TaskPredictions.hpp>
#include <TaskStatistics.hpp>
#include <TaskDataAccessesInfo.hpp>

struct DataAccess;
struct DataAccessBase;
struct StreamFunctionCallback;
class ComputePlace;
class MemoryPlace;
class WorkerThread;

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wunused-result"

using namespace ExecutionWorkflow;

class Task {
public:
	enum {
		//! Flags added by the Mercurium compiler
		final_flag=0,
		if0_flag,
		taskloop_flag,
		taskfor_flag,
		wait_flag,
		preallocated_args_block_flag,
		lint_verified_flag,
		//! Flags added by the Nanos6 runtime. Note that
		//! these flags must be always declared after the
		//! Mercurium flags
		non_runnable_flag,
		spawned_flag,
		remote_flag,
		stream_executor_flag,
		main_task_flag,
		total_flags
	};

	typedef long priority_t;

private:
	typedef std::bitset<total_flags> flags_t;

	void *_argsBlock;
	size_t _argsBlockSize;

	nanos6_task_info_t *_taskInfo;
	nanos6_task_invocation_info_t *_taskInvokationInfo;

	//! Number of children that are still not finished, +1 if not blocked
	std::atomic<int> _countdownToBeWokenUp;

	//! Number of children that are still alive (may have live references to data from this task), +1 for dependencies
	std::atomic<int> _removalCount;

	//! Task to which this one is closely nested
	Task *_parent;

	priority_t _priority;

protected:
	//! The thread assigned to this task, nullptr if the task has finished (but possibly waiting its children)
	std::atomic<WorkerThread *> _thread;

	//! Accesses that may determine dependencies
	TaskDataAccesses _dataAccesses;

	// Need to get back to the task from TaskDataAccesses for instrumentation purposes
	friend struct TaskDataAccesses;

	flags_t _flags;

private:
	//! Number of pending predecessors
	std::atomic<int> _predecessorCount;

	//! An identifier for the task for the instrumentation
	Instrument::task_id_t _instrumentationTaskId;

	//! Opaque data that is scheduler-dependent
	void *_schedulerInfo;

	//! Compute Place where the task is running
	ComputePlace *_computePlace;

	//! MemoryPlace "attached" to the ComputePlace the Task is running on
	MemoryPlace *_memoryPlace;

	//! Device Specific data
	void *_deviceData;

	//! Number of internal and external events that prevent the release of dependencies
	std::atomic<int> _countdownToRelease;

	//! Execution workflow to execute this Task
	Workflow<TaskExecutionWorkflowData> *_workflow;

	//! At the moment we will store the Execution step of the task
	//! here in order to invoke it after previous asynchronous
	//! steps have been completed.
	Step *_executionStep;

	//! Monitoring-related statistics about the task
	TaskStatistics _taskStatistics;

	//! Monitoring-related predictions about the task
	TaskPredictions _taskPredictions;

	//! Hardware counter structures of the task
	TaskHardwareCounters *_hwCounters;

	//! Cluster-related data for remote tasks
	TaskOffloading::ClusterTaskContext *_clusterContext;

	//! A pointer to the callback of the spawned function that created the
	//! task, used to trigger a callback from the appropriate stream function
	//! if the parent of this task is a StreamExecutor
	StreamFunctionCallback *_parentSpawnCallback;
public:
	inline Task(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		TaskDataAccessesInfo taskAccessInfo,
		void *taskCounters
	);

	virtual inline void reinitialize(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	);

	virtual inline ~Task();

	//! Set the address of the arguments block
	inline void setArgsBlock(void *argsBlock)
	{
		_argsBlock = argsBlock;
	}

	//! Get the address of the arguments block
	inline void *getArgsBlock() const
	{
		return _argsBlock;
	}
	//! Get the arguments block size
	inline size_t getArgsBlockSize() const
	{
		return _argsBlockSize;
	}
	inline void setArgsBlockSize(size_t argsBlockSize)
	{
		_argsBlockSize = argsBlockSize;
	}

	inline nanos6_task_info_t *getTaskInfo() const
	{
		return _taskInfo;
	}

	inline nanos6_task_invocation_info_t *getTaskInvokationInfo() const
	{
		return _taskInvokationInfo;
	}

	//! Actual code of the task
	virtual inline void body(void *deviceEnvironment, nanos6_address_translation_entry_t *translationTable = nullptr)
	{
		assert(_taskInfo->implementation_count == 1);
		assert(hasCode());
		assert(_taskInfo != nullptr);
		assert(!isTaskfor());
		_taskInfo->implementations[0].run(_argsBlock, deviceEnvironment, translationTable);
	}

	//! Check if the task has an actual body
	inline bool hasCode()
	{
		assert(_taskInfo->implementation_count == 1); // TODO: temporary check for a single implementation
		assert(_taskInfo != nullptr);
		return (_taskInfo->implementations[0].run != nullptr); // TODO: solution until multiple implementations are allowed
	}

	//! \brief sets the thread assigned to tun the task
	//!
	//! \param in thread the thread that will run the task
	inline void setThread(WorkerThread *thread)
	{
		assert(thread != nullptr);
		assert(_thread == nullptr);
		_thread = thread;
	}

	//! \brief get the thread that runs or will run the task
	//!
	//! \returns the thread that runs or will run the task
	inline WorkerThread *getThread() const
	{
		return _thread;
	}


	//! \brief Add a nested task
	inline void addChild(__attribute__((unused)) Task *child)
	{
		_countdownToBeWokenUp.fetch_add(1, std::memory_order_relaxed);
		_removalCount.fetch_add(1, std::memory_order_relaxed);
	}

	//! \brief Remove a nested task (because it has finished)
	//!
	//! \returns true iff the change makes this task become ready
	inline bool finishChild() __attribute__((warn_unused_result))
	{
		int countdown = (_countdownToBeWokenUp.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Remove a nested task (because it has been deleted)
	//!
	//! \returns true iff the change makes this task become disposable
	inline bool removeChild(__attribute__((unused)) Task *child) __attribute__((warn_unused_result))
	{
		int countdown = (_removalCount.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Increase an internal counter to prevent the removal of the task
	inline void increaseRemovalBlockingCount()
	{
		_removalCount.fetch_add(1, std::memory_order_relaxed);
	}

	//! \brief Decrease an internal counter that prevents the removal of the task
	//!
	//! \returns true iff the change makes this task become ready or disposable
	inline bool decreaseRemovalBlockingCount()
	{
		int countdown = (_removalCount.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Set the parent
	//! This should be used when the parent was not set during creation, and should have the parent in a state that allows
	//! adding this task as a child.
	//! \param parent inout the actual parent of the task
	inline void setParent(Task *parent)
	{
		assert(parent != nullptr);
		_parent = parent;
		_parent->addChild(this);
	}

	//! \brief Get the parent into which this task is nested
	//!
	//! \returns the task into which one is closely nested, or null if this is the main task
	inline Task *getParent() const
	{
		return _parent;
	}

	//! \brief Remove the link between the task and its parent
	//!
	//! \returns true iff the change made the parent become ready or disposable
	inline bool unlinkFromParent() __attribute__((warn_unused_result))
	{
		if (_parent != nullptr) {
			return _parent->removeChild(this);
		} else {
			return (_removalCount == 0);
		}
	}

	inline priority_t getPriority() const
	{
		return _priority;
	}

	inline void setPriority(priority_t priority)
	{
		_priority = priority;
	}

	//! \brief Mark that the task has finished its execution
	//! It marks the task as finished and determines if the
	//! dependencies can be released. The release could be
	//! postponed due to uncompleted external events. It could
	//! also be postponed due to a wait clause, in which the
	//! last child task should release the dependencies
	//!
	//! Note: This should be called only from the body of the
	//! thread that has executed the task
	//!
	//! \param computePlace in the compute place of the calling thread
	//!
	//! \returns true if its dependencies can be released
	inline bool markAsFinished(ComputePlace *computePlace);

	//! \brief Mark that the dependencies of the task have been released
	//!
	//! \returns true if the task can be disposed
	inline bool markAsReleased() __attribute__((warn_unused_result))
	{
		assert(_thread == nullptr);
		assert(_computePlace == nullptr);
		return decreaseRemovalBlockingCount();
	}

	//! \brief Mark that all its child tasks have finished
	//! It marks that all children have finished and determines
	//! if the dependencies can be released. It completes the
	//! delay of the dependency release in case the task has a
	//! wait clause, however, some external events could be still
	//! uncompleted
	//!
	//! Note: This should be called when unlinking the last child
	//! task (i.e. the removal counter becomes zero)
	//!
	//! \param computePlace in the compute place of the calling thread
	//!
	//! \returns true if its depedencies can be released
	inline bool markAllChildrenAsFinished(ComputePlace *computePlace);

	//! \brief Mark it as blocked
	//!
	//! \returns true if the change makes the task become ready
	inline bool markAsBlocked()
	{
		int countdown = (_countdownToBeWokenUp.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Mark it as unblocked
	//!
	//! \returns true if it does not have any children
	inline bool markAsUnblocked()
	{
		return (_countdownToBeWokenUp.fetch_add(1, std::memory_order_relaxed) == 0);
	}

	//! \brief Decrease the remaining count for unblocking the task
	//!
	//! \returns true if the change makes the task become ready
	inline bool decreaseBlockingCount()
	{
		int countdown = (_countdownToBeWokenUp.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Increase the remaining count for unblocking the task
	inline void increaseBlockingCount()
	{
		_countdownToBeWokenUp.fetch_add(1, std::memory_order_relaxed);
	}

	//! \brief Indicates whether it has finished
	inline bool hasFinished()
	{
		if (_taskInfo->implementations[0].device_type_id) {
			return (_computePlace == nullptr);
		} else {
			return (_thread == nullptr);
		}
	}

	//! \brief Indicates if it can be woken up
	//! Note: The task must have been marked as blocked
	inline bool canBeWokenUp()
	{
		// assert(_thread != nullptr);
		return (_countdownToBeWokenUp == 0);
	}

	//! \brief Indicates if it does not have any children
	inline bool doesNotNeedToBlockForChildren()
	{
		return (_removalCount == 1);
	}

	//! \brief Prevent this task to be scheduled when it is unblocked
	//!
	//! \returns true if this task is still not unblocked
	inline bool disableScheduling()
	{
		int countdown = _countdownToBeWokenUp.load();
		assert(countdown >= 0);

		// If it is 0 (unblocked), do not increment
		while (countdown > 0 && !_countdownToBeWokenUp.compare_exchange_strong(countdown, countdown + 1)) {
		}

		return (countdown > 0);
	}

	//! \brief Enable scheduling again for this task
	//!
	//! \returns true if this task is unblocked
	inline bool enableScheduling()
	{
		int countdown = (_countdownToBeWokenUp.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(countdown >= 0);
		return (countdown == 0);
	}

	//! \brief Retrieve the list of data accesses
	TaskDataAccesses const &getDataAccesses() const
	{
		return _dataAccesses;
	}

	//! \brief Retrieve the list of data accesses
	TaskDataAccesses &getDataAccesses()
	{
		return _dataAccesses;
	}

	//! \brief Increase the number of predecessors
	void increasePredecessors(int amount=1)
	{
		_predecessorCount += amount;
	}

	//! \brief Decrease the number of predecessors
	//! \returns true if the task becomes ready
	bool decreasePredecessors(int amount=1)
	{
		int res = (_predecessorCount-= amount);
		assert(res >= 0);
		return (res == 0);
	}

	//! \brief Set or unset the final flag
	void setFinal(bool finalValue)
	{
		_flags[final_flag] = finalValue;
	}
	//! \brief Check if the task is final
	bool isFinal() const
	{
		return _flags[final_flag];
	}

	//! \brief Set or unset the if0 flag
	void setIf0(bool if0Value)
	{
		_flags[if0_flag] = if0Value;
	}
	//! \brief Check if the task is in if0 mode
	bool isIf0() const
	{
		return _flags[if0_flag];
	}

	//! \brief Set or unset the taskloop flag
	void setTaskloop(bool taskloopValue)
	{
		_flags[taskloop_flag] = taskloopValue;
	}
	//! \brief Check if the task is a taskloop
	bool isTaskloop() const
	{
		return _flags[taskloop_flag];
	}

	//! \brief Set or unset the taskfor flag
	void setTaskfor(bool taskforValue)
	{
		_flags[taskfor_flag] = taskforValue;
	}
	//! \brief Check if the task is a taskfor
	bool isTaskfor() const
	{
		return _flags[taskfor_flag];
	}

	inline bool isRunnable() const
	{
		return !_flags[Task::non_runnable_flag];
	}

	//! \brief Set the wait behavior
	inline void setDelayedRelease(bool delayedReleaseValue)
	{
		_flags[Task::wait_flag] = delayedReleaseValue;
	}
	//! \brief Check if the task has the wait clause
	bool mustDelayRelease() const
	{
		return _flags[wait_flag];
	}

	//! \brief Complete the delay of the dependency release
	//! It completes the delay of the dependency release
	//! enforced by a wait clause
	void completeDelayedRelease()
	{
		assert(_flags[wait_flag]);
		_flags[wait_flag] = false;
	}

	bool hasPreallocatedArgsBlock() const
	{
		return _flags[preallocated_args_block_flag];
	}

	bool isSpawned() const
	{
		return _flags[spawned_flag];
	}

	void setSpawned(bool value=true)
	{
		_flags[spawned_flag] = value;
	}

	inline size_t getFlags() const
	{
		return _flags.to_ulong();
	}

	//! \brief Retrieve the instrumentation-specific task identifier
	inline Instrument::task_id_t getInstrumentationTaskId() const
	{
		return _instrumentationTaskId;
	}

	//! \brief Retrieve scheduler-dependent data
	inline void *getSchedulerInfo()
	{
		return _schedulerInfo;
	}

	//! \brief Set scheduler-dependent data
	inline void setSchedulerInfo(void *schedulerInfo)
	{
		_schedulerInfo = schedulerInfo;
	}

	//! \brief Increase the counter of events
	inline void increaseReleaseCount(int amount = 1)
	{
		_countdownToRelease += amount;
	}

	//! \brief Decrease the counter of events
	//!
	//! \returns true iff the dependencies can be released
	inline bool decreaseReleaseCount(int amount = 1)
	{
		int count = (_countdownToRelease -= amount);
		assert(count >= 0);
		return (count == 0);
	}

	//! \brief Return the number of symbols on the task
	inline int getNumSymbols()
	{
		return _taskInfo->num_symbols;
	}

	inline ComputePlace *getComputePlace() const
	{
		return _computePlace;
	}
	inline void setComputePlace(ComputePlace *computePlace)
	{
		_computePlace = computePlace;
	}
	inline bool hasComputePlace() const
	{
		return _computePlace != nullptr;
	}

	inline MemoryPlace *getMemoryPlace() const
	{
		return _memoryPlace;
	}
	inline void setMemoryPlace(MemoryPlace *memoryPlace)
	{
		_memoryPlace = memoryPlace;
	}
	inline bool hasMemoryPlace() const
	{
		return _memoryPlace != nullptr;
	}

	//! \brief Get the device type for which this task is implemented
	inline int getDeviceType()
	{
		return _taskInfo->implementations[0].device_type_id;
	}

	//! \brief Get the device subtype for which this task is implemented, TODO: device_subtype_id.
	inline int getDeviceSubType()
	{
		return 0;
	}

	inline void *getDeviceData()
	{
		return _deviceData;
	}
	inline void setDeviceData(void *deviceData)
	{
		_deviceData = deviceData;
	}

	//! \brief Set the Execution Workflow for this Task
	inline void setWorkflow(Workflow<TaskExecutionWorkflowData> *workflow)
	{
		assert(workflow != nullptr);
		_workflow = workflow;
	}
	//! \brief Get the Execution Workflow of the Task
	inline Workflow<TaskExecutionWorkflowData> *getWorkflow() const
	{
		return _workflow;
	}

	inline void setExecutionStep(ExecutionWorkflow::Step *step)
	{
		_executionStep = step;
	}
	inline ExecutionWorkflow::Step *getExecutionStep() const
	{
		return _executionStep;
	}

	//! \brief Get a label that identifies the tasktype
	inline const std::string getLabel() const
	{
		if (_taskInfo != nullptr) {
			if (_taskInfo->implementations != nullptr) {
				if (_taskInfo->implementations->task_label != nullptr) {
					return std::string(_taskInfo->implementations->task_label);
				} else if (_taskInfo->implementations->declaration_source != nullptr) {
					return std::string(_taskInfo->implementations->declaration_source);
				}
			}

			// If the label is empty, use the invocation source
			return std::string(_taskInvokationInfo->invocation_source);
		} else if (_parent != nullptr) {
			return _parent->getLabel();
		}

		return "Unlabeled";
	}

	//! \brief Check whether cost is available for the task
	inline bool hasCost() const
	{
		if (_taskInfo->implementations != nullptr) {
			return (_taskInfo->implementations->get_constraints != nullptr);
		}
		else {
			return false;
		}
	}

	//! \brief Get the task's cost
	inline size_t getCost() const
	{
		assert(_taskInfo->implementations != nullptr);

		nanos6_task_constraints_t constraints;
		_taskInfo->implementations->get_constraints(_argsBlock, &constraints);

		return constraints.cost;
	}

	//! \brief Get the task's statistics
	inline TaskStatistics *getTaskStatistics()
	{
		return &_taskStatistics;
	}

	//! \brief Get the task's predictions
	inline TaskPredictions *getTaskPredictions()
	{
		return &_taskPredictions;
	}

	//! \brief Setter for the task's hardware counter structures
	inline void setHardwareCounters(TaskHardwareCounters *hwCounters)
	{
		_hwCounters = hwCounters;
	}

	//! \brief Get the task's hardware counter structures
	inline TaskHardwareCounters *getHardwareCounters()
	{
		return _hwCounters;
	}

	inline void markAsRemote()
	{
		_flags[remote_flag] = true;
	}
	inline bool isRemote() const
	{
		return _flags[remote_flag];
	}

	inline void setClusterContext(
		TaskOffloading::ClusterTaskContext *clusterContext)
	{
		_clusterContext = clusterContext;
	}

	inline TaskOffloading::ClusterTaskContext *getClusterContext() const
	{
		return _clusterContext;
	}

	inline void markAsStreamExecutor()
	{
		_flags[stream_executor_flag] = true;
	}

	inline bool isStreamExecutor() const
	{
		return _flags[stream_executor_flag];
	}

	inline void setParentSpawnCallback(StreamFunctionCallback *callback)
	{
		_parentSpawnCallback = callback;
	}

	inline StreamFunctionCallback *getParentSpawnCallback() const
	{
		return _parentSpawnCallback;
	}

	inline void markAsMainTask()
	{
		_flags[main_task_flag] = true;
	}

	inline bool isMainTask() const
	{
		return _flags[main_task_flag];
	}
};


#pragma GCC diagnostic push


#endif // TASK_HPP

