#ifndef TASK_HPP
#define TASK_HPP


#include <atomic>
#include <cassert>
#include <set>

#include "lowlevel/SpinLock.hpp"


class WorkerThread;


#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wunused-result"


class Task {
	//! The thread assigned to this task, nullptr if the task has finished (but possibly waiting its children)
	std::atomic<WorkerThread *> _thread;
	
	//! Number of children that are still alive (may have live references to data from this task), +1 if not blocked
	std::atomic<int> _countdownToBeWokenUp;
	
	//! Task to which this one is closely nested
	Task *_parent;
	
public:
	Task(Task *parent)
		: _thread(nullptr), _countdownToBeWokenUp(1), _parent(parent)
	{
		if (parent != nullptr) {
			addChild(this);
		}
	}
	
	virtual ~Task()
	{
	}
	
	
	//! Actual code of the task
	virtual void body() = 0;
	
	
	//! \brief sets the thread assigned to tun the task
	//!
	//! \param in thread the thread that will run the task
	inline void setThread(WorkerThread *thread)
	{
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
	
	//! \brief clears the thread assigned to this task
	inline void unsetThread()
	{
		_thread = nullptr;
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
		return ((--_countdownToBeWokenUp) == 0);
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
			return false;
		}
	}
	
	
	//! \brief Mark it as finished
	//!
	//! \returns true if the change makes the task disposable
	inline bool markAsFinished() __attribute__((warn_unused_result))
	{
		_thread = nullptr;
		return ((--_countdownToBeWokenUp) == 0);
	}
	
	//! \brief Mark it as blocked
	//!
	//! \returns true if the change makes the task become ready
	inline bool markAsBlocked()
	{
		return ((--_countdownToBeWokenUp) == 0);
	}
	
	//! \brief Mark it as unblocked
	//!
	//! \returns true if it does not have any children
	inline bool markAsUnblocked()
	{
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
		return (_countdownToBeWokenUp == 0);
	}
	
	//! \brief Indicates if it does not have any children (while unblocked)
	//! Note: The task must not be blocked
	inline bool doesNotNeedToBlockForChildren()
	{
		return (_countdownToBeWokenUp == 1);
	}
	
};


#pragma GCC diagnostic push


#endif // TASK_HPP

