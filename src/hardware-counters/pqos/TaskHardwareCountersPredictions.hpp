/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_PREDICTIONS_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_PREDICTIONS_HPP

#include "hardware-counters/SupportedHardwareCounters.hpp"


class TasktypeHardwareCountersPredictions;

struct CounterPrediction {
	
	// Whether a prediction is available for this counter
	bool _predictionAvailable;
	
	// The prediction for the counter
	double _prediction;
	
	
	inline CounterPrediction() :
		_predictionAvailable(false)
	{
	}
};


class TaskHardwareCountersPredictions {

private:
	
	//! Task-specific PQoS counter predictions
	CounterPrediction _counterPredictions[HWCounters::num_counters];
	
	//! Predictions for counter values of the tasktype
	TasktypeHardwareCountersPredictions *_typePredictions;
	
	
public:
	
	inline TaskHardwareCountersPredictions() :
		_typePredictions(nullptr)
	{
	}
	
	
	//! \brief Set whether a counter has a prediction
	//! \param counterId The counter's id
	//! \param available Whether a prediction is available
	inline void setPredictionAvailable(HWCounters::counters_t counterId, bool available)
	{
		_counterPredictions[counterId]._predictionAvailable = available;
	}
	
	//! \brief Check whether a counter has a prediction
	//! \param counterId The counter's id
	inline bool hasPrediction(HWCounters::counters_t counterId) const
	{
		return _counterPredictions[counterId]._predictionAvailable;
	}
	
	//! \brief Set the prediction for a counter
	//! \param counterId The counter's id
	//! \param value The value for the prediction of the counter
	inline void setCounterPrediction(HWCounters::counters_t counterId, double value)
	{
		_counterPredictions[counterId]._prediction = value;
	}
	
	//! \brief Get the prediction for a certain counter
	//! \param counterId The counter's id
	inline double getCounterPrediction(HWCounters::counters_t counterId) const
	{
		return _counterPredictions[counterId]._prediction;
	}
	
	//! \brief Set the reference for tasktype predictions
	inline void setTypePredictions(TasktypeHardwareCountersPredictions *predictions)
	{
		_typePredictions = predictions;
	}
	
	//! \brief Get the reference for tasktype predictions
	inline TasktypeHardwareCountersPredictions *getTypePredictions() const
	{
		return _typePredictions;
	}
	
};

#endif // PQOS_TASK_HARDWARE_COUNTERS_PREDICTIONS_HPP
