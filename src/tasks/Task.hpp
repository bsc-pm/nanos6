/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HPP
#define TASK_HPP


#include <atomic>
#include <bitset>
#include <cassert>
#include <set>

#include <nanos6.h>
#include "lowlevel/SpinLock.hpp"

#include <InstrumentTaskId.hpp>

#include <TaskDataAccesses.hpp>


struct DataAccess;
struct DataAccessBase;
class WorkerThread;
class ComputePlace;

#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wunused-result"


class Task {
public:
	enum {
		final_flag=0,
		if0_flag,
		taskloop_flag,
		wait_flag,
		non_runnable_flag, // Note: Must be at the end
		total_flags
	};
	
	typedef long priority_t;
	
private:
	typedef std::bitset<total_flags> flags_t;
	
	void *_argsBlock;
	
	nanos6_task_info *_taskInfo;
	nanos6_task_invocation_info *_taskInvokationInfo;
	
	//! Number of children that are still alive (may have live references to data from this task), +1 if not blocked
	std::atomic<int> _countdownToBeWokenUp;
	
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
	
	//! Device Specific data
	void *_deviceData;
	
	//! Number of internal and external events that prevent the release of dependencies
	std::atomic<int> _countdownToRelease;
	
public:
	inline Task(
		void *argsBlock,
		nanos6_task_info *taskInfo,
		nanos6_task_invocation_info *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	);
	
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
	
	inline nanos6_task_info *getTaskInfo() const
	{
		return _taskInfo;
	}
	
	inline nanos6_task_invocation_info *getTaskInvokationInfo() const
	{
		return _taskInvokationInfo;
	}
	
	//! Actual code of the task
	virtual inline void body(void *deviceEnvironment, nanos6_address_translation_entry_t *translationTable = nullptr)
	{
		assert(_taskInfo->implementation_count == 1);
		assert(hasCode());
		assert(_taskInfo != nullptr);	
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
		++_countdownToBeWokenUp;
	}
	
	//! \brief Remove a nested task (because it has finished)
	//!
	//! \returns true iff the change makes this task become ready or disposable
	inline bool removeChild(__attribute__((unused)) Task *child) __attribute__((warn_unused_result))
	{
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return (countdown == 0);
	}
	
	//! \brief Increase an internal counter to prevent the removal of the task
	inline void increaseRemovalBlockingCount()
	{
		++_countdownToBeWokenUp;
	}
	
	//! \brief Decrease an internal counter that prevents the removal of the task
	//!
	//! \returns true iff the change makes this task become ready or disposable
	inline bool decreaseRemovalBlockingCount()
	{
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return (countdown == 0);
	}
	
	//! \brief Decrease an internal counter that prevents the removal of the task
	//!
	//! \returns the counter's value after decreasing it
	inline int decreaseAndGetRemovalBlockingCount()
	{
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return countdown;
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
			return (_countdownToBeWokenUp == 0);
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
		assert(_thread != nullptr);
		
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return (countdown == 0);
	}
	
	//! \brief Mark it as unblocked
	//!
	//! \returns true if it does not have any children
	inline bool markAsUnblocked()
	{
		assert(_thread != nullptr);
		return ((++_countdownToBeWokenUp) == 1);
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
		assert(_thread != nullptr);
		return (_countdownToBeWokenUp == 0);
	}
	
	//! \brief Indicates if it does not have any children (while unblocked)
	//! Note: The task must not be blocked
	inline bool doesNotNeedToBlockForChildren()
	{
		return (_countdownToBeWokenUp == 1);
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
	
	//! \brief Enable shceduling again for this task
	//!
	//! \returns true if this task is unblocked
	inline bool enableScheduling()
	{
		int countdown = (--_countdownToBeWokenUp);
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
		return ((_predecessorCount -= amount) == 0);
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
	inline int getSymbolNum(){
		return _taskInfo->num_symbols;
	}
	
	inline ComputePlace *getComputePlace()
	{
		return _computePlace;
	}
	inline void setComputePlace(ComputePlace *computePlace){
		_computePlace = computePlace;
	}
	
	//! \brief Get the device type for which this task is implemented
	inline int getDeviceType()
	{
		return _taskInfo->implementations[0].device_type_id;
	}
	
	inline void *getDeviceData()
	{
		return _deviceData;
	}
	inline void setDeviceData(void *deviceData)
	{
		_deviceData = deviceData;	
	}
};


#pragma GCC diagnostic push


#endif // TASK_HPP

