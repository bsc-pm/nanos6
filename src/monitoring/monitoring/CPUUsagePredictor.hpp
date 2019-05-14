/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_USAGE_PREDICTOR_HPP
#define CPU_USAGE_PREDICTOR_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "nanos6/polling.h"

#include "CPUMonitor.hpp"
#include "lowlevel/EnvironmentVariable.hpp"


class CPUUsagePredictor {

private:
	typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::sum, boost::accumulators::tag::mean> > accumulator_t;
	
	//! The predictor singleton instance
	static CPUUsagePredictor *_predictor;
	
	//! How often (microseconds) CPU utilization predictions are made
	static EnvironmentVariable<size_t> _predictionRate;
	
	//! The current prediction for CPU usage (number of CPUs to use)
	double _prediction;
	
	//! Whether the current prediction is available
	bool _predictionAvailable;
	
	//! Accumulator of prediction accuracies
	accumulator_t _accuracies;
	
	//! A chronometer to issue predictions under demand
	Chrono _predictionFrequency;
	
	
private:
	
	inline CPUUsagePredictor() :
		_predictionAvailable(false),
		_accuracies(),
		_predictionFrequency()
	{
	}
	
	
	//! \brief Service routine that predicts the CPU usage at a certain rate
	//! \return Whether the service routine should be stopped
	static inline int predictCPUUsageService(void *)
	{
		assert(_predictor != nullptr);
		
		// Check the CPU usage has to be predicted
		_predictor->_predictionFrequency.stop();
		if (((double) _predictor->_predictionFrequency) >= (double) _predictionRate) {
			_predictor->_predictionAvailable = true;
			_predictor->_predictionFrequency.restart();
			_predictor->_prediction = getPredictedCPUUsage(_predictionRate.getValue());
		}
		_predictor->_predictionFrequency.start();
		
		return false;
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	CPUUsagePredictor(CPUUsagePredictor const&) = delete;            // Copy construct
	CPUUsagePredictor(CPUUsagePredictor&&) = delete;                 // Move construct
	CPUUsagePredictor& operator=(CPUUsagePredictor const&) = delete; // Copy assign
	CPUUsagePredictor& operator=(CPUUsagePredictor &&) = delete;     // Move assign
	
	
	//! \brief Initialize the CPU usage predictor
	static inline void initialize()
	{
		// Create the singleton
		if (_predictor == nullptr) {
			_predictor = new CPUUsagePredictor();
			
			// Start the timer that measures the frequency of issuing predictions
			_predictor->_predictionFrequency.start();
			
			// Register a service that predicts CPU usage at a frequency of time
			nanos6_register_polling_service(
				"CPUUsagePredictionService",
				predictCPUUsageService,
				nullptr
			);
		}
	}
	
	//! \brief Shutdown the CPU usage predictor
	static inline void shutdown()
	{
		if (_predictor != nullptr) {
			// Unregister the service that predicts CPU usage
			nanos6_unregister_polling_service(
				"CPUUsagePredictionService",
				predictCPUUsageService,
				nullptr
			);
			
			// Destroy the monitoring module
			delete _predictor;
		}
	}
	
	//! \brief Display CPU usage predictions' statistics
	//! \param stream The output stream
	static inline void displayStatistics(std::stringstream &stream)
	{
		assert(_predictor != nullptr);
		
		stream << std::left << std::fixed << std::setprecision(2) << "\n";
		stream << "+-----------------------------+\n";
		stream << "|    CPU Usage Predictions    |\n";
		stream << "+-----------------------------+\n";
		stream << "  MEAN ACCURACY: " << boost::accumulators::mean(_predictor->_accuracies) << "%\n";
		stream << "+-----------------------------+\n\n";
	}
	
	//! \brief Get a prediction for the amount of CPU usage to use
	//! taking into account the current workload in the runtime
	//! \param[in] time The amount of time in microseconds to predict usage for
	//! (i.e. in the next 'time' microseconds, the amount of CPUs to be used)
	static double getPredictedCPUUsage(size_t time);
	
};

#endif // CPU_USAGE_PREDICTOR_HPP

