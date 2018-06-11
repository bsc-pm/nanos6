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
		non_runnable_flag, // NOTE: Must be at the end
		delayed_release_flag, // NOTE: Must be at the end
		total_flags
	};
	
	typedef long priority_t;
	
private:
	typedef std::bitset<total_flags> flags_t;
	
	void *_argsBlock;
	
	nanos_task_info *_taskInfo;
	nanos_task_invocation_info *_taskInvokationInfo;
	
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
	
public:
	inline Task(
		void *argsBlock,
		nanos_task_info *taskInfo,
		nanos_task_invocation_info *taskInvokationInfo,
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
	
	inline nanos_task_info *getTaskInfo() const
	{
		return _taskInfo;
	}
	
	inline nanos_task_invocation_info *getTaskInvokationInfo() const
	{
		return _taskInvokationInfo;
	}
	
	//! Actual code of the task
	virtual inline void body(void *deviceEnvironment)
	{
		assert(_taskInfo->implementation_count == 1);
		assert(hasCode());
		assert(_taskInfo != nullptr);	
		_taskInfo->implementations[0].run(_argsBlock, deviceEnvironment, nullptr);
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
	
	
	//! \brief Mark it as finished
	//!
	//! \returns true if the change makes the task disposable
	virtual inline bool markAsFinished() __attribute__((warn_unused_result))
	{
		if (_taskInfo->implementations[0].device_type_id == nanos6_device_t::nanos6_host_device) {
			assert(_thread != nullptr);
			_thread = nullptr;
		} else {
			assert(_computePlace != nullptr);
			_computePlace = nullptr;
		}
		
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return (countdown == 0);
	}
	
	//! \brief Mark it as finished after the data access release
	//!
	//! \returns true if the change makes the task disposable
	inline bool markAsFinishedAfterDataAccessRelease() __attribute__((warn_unused_result))
	{
		if( _taskInfo->implementations[0].device_type_id == nanos6_device_t::nanos6_host_device) {
			if (hasDelayedDataAccessRelease()) {
				assert(_thread == nullptr);
				setDelayedDataAccessRelease(false);
			} else {
				assert(_thread != nullptr);
				_thread = nullptr;
			}
		} else {
			if (hasDelayedDataAccessRelease()) {
				assert(_computePlace == nullptr);
				setDelayedDataAccessRelease(false);
			} else {
				assert(_computePlace != nullptr);
				_computePlace = nullptr;
			}
		}
		
		int countdown = (--_countdownToBeWokenUp);
		assert(countdown >= 0);
		return (countdown == 0);
	}
	
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
	
	//! \brief Set or unset the wait flag
	void setDelayDataAccessRelease(bool delayValue)
	{
		_flags[wait_flag] = delayValue;
	}
	//! \brief Check if the task has the wait clause
	bool mustDelayDataAccessRelease() const
	{
		return _flags[wait_flag];
	}
	
	//! \brief Set or unset the delayed release flag
	void setDelayedDataAccessRelease(bool delayedValue)
	{
		_flags[delayed_release_flag] = delayedValue;
	}
	
	//! \brief Check if the task has delayed the data access release
	bool hasDelayedDataAccessRelease() const
	{
		return _flags[delayed_release_flag];
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

