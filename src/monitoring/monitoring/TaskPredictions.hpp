/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_PREDICTIONS_HPP
#define TASK_PREDICTIONS_HPP

#include <atomic>


class TasktypePredictions;

class TaskPredictions {

private:
	
	//! Whether a timing prediction is available for the task
	bool _taskHasPrediction;
	
	//! Predicted elapsed execution time of the task
	double _timePrediction;
	
	//! A pointer to the TaskPredictions of the parent task
	TaskPredictions *_parentPredictions;
	
	//! Whether a task in the chain of ancestors of this task had predictions
	bool _ancestorHasPrediction;
	
	//! Ticks of children tasks, used to obtain more accurate predictions
	//! (not converted to time, simply internal chronometer ticks)
	std::atomic<size_t> _childCompletionTimes;
	
	//! A pointer to the accumulated predictions of the tasktype
	TasktypePredictions *_typePredictions;
	
	
public:
	
	inline TaskPredictions() :
		_taskHasPrediction(false),
		_parentPredictions(nullptr),
		_ancestorHasPrediction(false),
		_childCompletionTimes(0),
		_typePredictions(nullptr)
	{
	}
	
	
	inline void setHasPrediction(bool hasPrediction)
	{
		_taskHasPrediction = hasPrediction;
	}
	
	inline bool hasPrediction() const
	{
		return _taskHasPrediction;
	}
	
	inline void setTimePrediction(double prediction)
	{
		_timePrediction = prediction;
	}
	
	inline double getTimePrediction() const
	{
		return _timePrediction;
	}
	
	inline void setParentPredictions(TaskPredictions *parentPredictions)
	{
		_parentPredictions = parentPredictions;
	}
	
	inline TaskPredictions *getParentPredictions() const
	{
		return _parentPredictions;
	}
	
	inline void setAncestorHasPrediction(bool ancestorHasPrediction)
	{
		_ancestorHasPrediction = ancestorHasPrediction;
	}
	
	inline bool ancestorHasPrediction() const
	{
		return _ancestorHasPrediction;
	}
	
	inline size_t getChildCompletionTimes() const
	{
		return _childCompletionTimes.load();
	}
	
	inline void increaseChildCompletionTimes(size_t elapsed)
	{
		_childCompletionTimes += elapsed;
	}
	
	inline void setTypePredictions(TasktypePredictions *predictions)
	{
		_typePredictions = predictions;
	}
	
	inline TasktypePredictions *getTypePredictions() const
	{
		return _typePredictions;
	}
	
};

#endif // TASK_PREDICTIONS_HPP
