/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HPP
#define TASK_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <string>

#include <nanos6.h>

#include "hardware/device/DeviceEnvironment.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "lowlevel/SpinLock.hpp"
#include "scheduling/ReadyQueue.hpp"

#include <InstrumentTaskId.hpp>
#include <TaskDataAccesses.hpp>
#include <TaskDataAccessesInfo.hpp>

struct DataAccess;
struct DataAccessBase;
struct StreamFunctionCallback;
class ComputePlace;
class MemoryPlace;
class TaskStatistics;
class TaskInfoData;
class WorkerThread;

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wunused-result"


class Task {
public:
	enum {
		//! Flags added by the Mercurium compiler
		final_flag=0,
		if0_flag,
		taskloop_flag,
		//! Taskfors are no longer supported. Keep this flag
		//! because the compiler can still generate taskfors.
		//! We treat taskfors as normal tasks.
		taskfor_flag,
		wait_flag,
		preallocated_args_block_flag,
		lint_verified_flag,
		taskiter_flag,
		taskiter_update_flag,
		//! Flags added by the Nanos6 runtime. Note that
		//! these flags must be always declared after the
		//! Mercurium flags
		non_runnable_flag,
		spawned_flag,
		remote_flag,
		stream_executor_flag,
		main_task_flag,
		onready_completed_flag,
		total_flags
	};

	typedef long priority_t;

	typedef uint64_t deadline_t;

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

	//! Task priority
	priority_t _priority;

	//! Task deadline to start/resume in microseconds (zero by default)
	deadline_t _deadline;

	//! Scheduling hint used by the scheduler
	ReadyTaskHint _schedulingHint;

	//! NUMA Locality scheduling hints
	uint64_t _NUMAHint;

protected:
	//! The thread assigned to this task, nullptr if the task has finished (but possibly waiting its children)
	std::atomic<WorkerThread *> _thread;

	//! Accesses that may determine dependencies
	TaskDataAccesses _dataAccesses;

	// Need to get back to the task from TaskDataAccesses for instrumentation purposes
	friend struct TaskDataAccesses;

	//! Task flags
	flags_t _flags;

private:
	//! Number of pending predecessors
	std::atomic<int> _predecessorCount;

	//! An identifier for the task for the instrumentation
	Instrument::task_id_t _instrumentationTaskId;

	//! Compute Place where the task is running
	ComputePlace *_computePlace;

	//! MemoryPlace "attached" to the ComputePlace the Task is running on
	MemoryPlace *_memoryPlace;

	//! Device Specific data
	void *_deviceData;

	//! Device Environment
	DeviceEnvironment _deviceEnvironment;

	//! Number of internal and external events that prevent the release of dependencies
	std::atomic<int> _countdownToRelease;

	//! Monitoring-related statistics about the task
	TaskStatistics *_taskStatistics;

	//! Hardware counter structures of the task
	TaskHardwareCounters _hwCounters;

	//! A pointer to the callback of the spawned function that created the
	//! task, used to trigger a callback from the appropriate stream function
	//! if the parent of this task is a StreamExecutor
	StreamFunctionCallback *_parentSpawnCallback;

	//! Nesting level of the task
	int _nestingLevel;
public:
	inline Task(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		const TaskDataAccessesInfo &taskAccessInfo,
		void *taskCountersAddress,
		void *taskStatistics
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
	virtual inline void body(nanos6_address_translation_entry_t *translationTable = nullptr)
	{
		assert(_taskInfo->implementation_count == 1);
		assert(hasCode());
		assert(_taskInfo != nullptr);

		_taskInfo->implementations[0].run(_argsBlock, (void *)&_deviceEnvironment, translationTable);
	}

	//! Check if the task has an actual body
	inline bool hasCode() const
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
		_nestingLevel = _parent->getNestingLevel() + 1;
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

	//! \brief Get the task priority
	//!
	//! \returns the priority
	inline priority_t getPriority() const
	{
		return _priority;
	}

	//! \brief Compute the task priority defined by the user
	//!
	//! \returns whether the task has a user-defined priority
	inline bool computePriority()
	{
		assert(_taskInfo != nullptr);
		assert(_argsBlock != nullptr);

		if (_taskInfo->get_priority != nullptr) {
			_taskInfo->get_priority(_argsBlock, &_priority);
			return true;
		}
		// Leave the default priority
		return false;
	}

	//! \brief Indicates whether the task has deadline
	//!
	//! \returns whether the task has deadline
	inline bool hasDeadline() const
	{
		return (_deadline > 0);
	}

	//! \brief Get the task deadline (us) to start/resume
	//!
	//! \returns the task deadline in us
	inline deadline_t getDeadline() const
	{
		return _deadline;
	}

	//! \brief Set the task deadline (us) to start/resume
	//!
	//! \param deadline the new task deadline in us
	inline void setDeadline(deadline_t deadline)
	{
		_deadline = deadline;
	}

	//! \brief Get the task scheduling hint
	//!
	//! \returns the scheduling hint
	inline ReadyTaskHint getSchedulingHint() const
	{
		return _schedulingHint;
	}

	//! \brief Set the task scheduling hint
	//!
	//! \param hint the new scheduling hint
	inline void setSchedulingHint(ReadyTaskHint hint)
	{
		_schedulingHint = hint;
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
		if (_taskInfo->implementations[0].device_type_id != nanos6_host_device) {
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

	inline int getPendingChildTasks() const
	{
		return _countdownToBeWokenUp.load(std::memory_order_relaxed) - 1;
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

	//! \brief Check if the task has an onready action
	inline bool hasOnready() const
	{
		assert(_taskInfo != nullptr);

		return (_taskInfo->onready_action != nullptr);
	}

	//! \brief Perform the onready action if needed
	//!
	//! This function executes the onready if needed and returns whether
	//! the task is ready to be executed (scheduled). If the task has no
	//! onready action or the onready do not bind any external event, the
	//! onready phase is automatically completed; otherwise the onready
	//! phase will have to be manually completed once all onready events
	//! finalize (see completeOnready function)
	//!
	//! \param currentThread the current thread
	//!
	//! \returns whether the task is ready to execute
	bool handleOnready(WorkerThread *currentThread)
	{
		assert(_predecessorCount == 0);

		if (isOnreadyCompleted()) {
			return true;
		}

		// Run onready action if present
		if (hasOnready()) {
			runOnready(currentThread);

			// Check whether has pending events
			if (!decreaseReleaseCount()) {
				// The execution was delayed due to onready events
				return false;
			}

			// Reset the counter before executing the task
			resetReleaseCount();
		}

		// Set the onready flag as completed
		setCompletedOnready();

		// The task is ready to execute
		return true;
	}

	//! \brief Complete the onready stage
	//!
	//! This function should be called once all events registered
	//! during the onready phase have finalized. After this call,
	//! the task is ready to be executed (scheduled)
	void completeOnready()
	{
		// Set the onready flag as completed
		setCompletedOnready();

		// Reset the event counter before executing the task
		resetReleaseCount();
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

	//! \brief Reset the counter of events
	inline void resetReleaseCount()
	{
		assert(_countdownToRelease == 0);

		_countdownToRelease = 1;
	}

	//! \brief Increase the counter of events
	inline void increaseReleaseCount(int amount = 1)
	{
		assert(_countdownToRelease > 0);

		_countdownToRelease += amount;
	}

	//! \brief Decrease the counter of events
	//!
	//! This function returns whether the decreased events were
	//! the last ones. This may mean that the task can start
	//! running if they were onready events or the task can release
	//! its dependencies if they were normal events
	//!
	//! \returns true iff were the last events
	inline bool decreaseReleaseCount(int amount = 1)
	{
		int count = (_countdownToRelease -= amount);
		assert(count >= 0);
		return (count == 0);
	}

	//! \brief Get the number of events
	//!
	//! This function returns the number of events of this task. The number
	//! of external events may change at any point if any is decreased from
	//! another thread
	//!
	//! \returns the number of external events
	inline int getReleaseCount()
	{
		return _countdownToRelease.load(std::memory_order_relaxed);
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

	inline DeviceEnvironment &getDeviceEnvironment()
	{
		return _deviceEnvironment;
	}

	//! \brief Get a label that identifies the tasktype
	inline const std::string getLabel() const
	{
		if (_taskInfo != nullptr) {
			if (_taskInfo->implementations != nullptr) {
				if (_taskInfo->implementations->task_type_label != nullptr) {
					return std::string(_taskInfo->implementations->task_type_label);
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
		if (_taskInfo != nullptr) {
			if (_taskInfo->implementations != nullptr) {
				return (_taskInfo->implementations->get_constraints != nullptr);
			}
		}

		return false;
	}

	//! \brief Get the task's cost
	inline size_t getCost() const
	{
		size_t cost = 1;
		if (hasCost()) {
			nanos6_task_constraints_t constraints;
			_taskInfo->implementations->get_constraints(_argsBlock, &constraints);
			cost = constraints.cost;
		}

		return cost;
	}

	//! \brief Get the task's monitoring statistics
	inline TaskStatistics *getTaskStatistics()
	{
		return _taskStatistics;
	}

	//! \brief Get the task's hardware counter structures
	inline TaskHardwareCounters &getHardwareCounters()
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

	inline bool isOnreadyCompleted() const
	{
		return _flags[onready_completed_flag];
	}

	inline TaskInfoData *getTaskInfoData() const
	{
		if (_taskInfo != nullptr) {
			return (TaskInfoData *) _taskInfo->task_type_data;
		}
		return nullptr;
	}

	virtual inline void registerDependencies(bool = false)
	{
		_taskInfo->register_depinfo(_argsBlock, nullptr, this);
	}

	virtual inline bool isDisposable() const
	{
		return true;
	}

	virtual inline bool isTaskloopSource() const
	{
		return false;
	}

	inline int getNestingLevel() const
	{
		return _nestingLevel;
	}

	virtual inline void increaseMaxChildDependencies()
	{
	}

	inline void computeNUMAAffinity(ComputePlace *computePlace)
	{
		_NUMAHint = _dataAccesses.computeNUMAAffinity(computePlace);
	}

	inline uint64_t getNUMAHint() const
	{
		return _NUMAHint;
	}

private:
	//! \brief Set the onready completed flag
	inline void setCompletedOnready()
	{
		_flags[onready_completed_flag] = true;
	}

	//! \brief Run the onready action
	//!
	//! \param currentThread the current thread
	void runOnready(WorkerThread *currentThread);
};


#pragma GCC diagnostic push


#endif // TASK_HPP

