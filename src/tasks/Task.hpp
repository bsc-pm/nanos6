#ifndef TASK_HPP
#define TASK_HPP


#include <atomic>
#include <bitset>
#include <cassert>
#include <set>

#include "api/nanos6_rt_interface.h"
#include "lowlevel/SpinLock.hpp"

#include <InstrumentTaskId.hpp>
#include <TaskDataAccesses.hpp>
#include <TaskDataAccessLinkingArtifacts.hpp>


struct DataAccess;
struct DataAccessBase;
class WorkerThread;
class GenericCache;


#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wunused-result"


class Task {
private:
	enum {
		final_flag=0,
		total_flags
	};
	typedef std::bitset<total_flags> flags_t;
	
	void *_argsBlock;
	
	nanos_task_info *_taskInfo;
	nanos_task_invocation_info *_taskInvokationInfo;
	
	//! The thread assigned to this task, nullptr if the task has finished (but possibly waiting its children)
	std::atomic<WorkerThread *> _thread;
	
	//! Number of children that are still alive (may have live references to data from this task), +1 if not blocked
	std::atomic<int> _countdownToBeWokenUp;
	
	//! Task to which this one is closely nested
	Task *_parent;
	
protected:
	//! Accesses that may determine dependencies
	TaskDataAccesses _dataAccesses;
	
	// Need to get back to the task from TaskDataAccesses for instrumentation purposes
	friend struct TaskDataAccesses;
	
private:
	//! Number of pending predecessors
	std::atomic<int> _predecessorCount;
	
	flags_t _flags;
	
	//! An identifier for the task for the instrumentation
	Instrument::task_id_t _instrumentationTaskId;
	
	//! Opaque data that is scheduler-dependent
	void *_schedulerInfo;
    
    //! Cache info
	bool _enabledCopies;
    std::size_t _taskDataSize;
    std::size_t _cachedBytes;
    GenericCache * _cache;
	
public:
	Task(
		void *argsBlock,
		nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	)
		: _argsBlock(argsBlock),
		_taskInfo(taskInfo), _taskInvokationInfo(taskInvokationInfo),
		_thread(nullptr), _countdownToBeWokenUp(1),
		_parent(parent),
		_dataAccesses(),
		_predecessorCount(0),
		_flags(flags),
		_instrumentationTaskId(instrumentationTaskId),
		_schedulerInfo(nullptr),
		_enabledCopies(false),
        _taskDataSize(0),
        _cachedBytes(0),
        _cache(nullptr)
	{
		if (parent != nullptr) {
			parent->addChild(this);
		}
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
	inline void body()
	{
		assert(_taskInfo != nullptr);
		_taskInfo->run(_argsBlock);
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
	
	
	//! \brief Mark it as finished
	//!
	//! \returns true if the change makes the task disposable
	inline bool markAsFinished() __attribute__((warn_unused_result))
	{
		assert(_thread != nullptr);
		_thread = nullptr;
		
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
		return (_thread == nullptr);
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
	//! \brief Check if tha task is final
	bool isFinal() const
	{
		return _flags[final_flag];
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
	

    //! \brief Update _cachedBytes with the amount specified and return the new value
    inline std::size_t addCachedBytes(std::size_t cachedBytes) {
        _cachedBytes += cachedBytes;
        return _cachedBytes;
    }

    //! \brief Set _cachedBytes to the amount specified
    inline void setCachedBytes(std::size_t cachedBytes) {
        _cachedBytes = cachedBytes;
    }

    //! \brief Return the current number of bytes that a task has in the cache where it is going to be executed.
    inline unsigned int getCachedBytes() {
        return _cachedBytes;
    }

    //! \brief Update _taskDataSize with the amount specified
    inline void addDataSize(std::size_t size) {
        _taskDataSize += size;
    }

    //! \brief Return the total number of bytes required by a task to be executed
    inline unsigned int getDataSize() const
    {
        return _taskDataSize;
    }

    inline bool hasPendingCopies() const 
    {
        return _cachedBytes < _taskDataSize;
    }

    inline void setCache(GenericCache * cache) {
        _cache = cache;
    }

    inline GenericCache * getCache() const {
        return _cache;
    }
	
	inline bool hasEnabledCopies()
	{
		return _enabledCopies;
	}

	inline void setEnabledCopies(bool enabled)
	{
		_enabledCopies = enabled;
	}
};


#pragma GCC diagnostic push


#include <TaskDataAccessLinkingArtifactsImplementation.hpp>


#endif // TASK_HPP

