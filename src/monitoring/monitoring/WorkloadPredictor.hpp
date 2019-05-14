/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKLOAD_PREDICTOR_HPP
#define WORKLOAD_PREDICTOR_HPP

#include <iomanip>
#include <iostream>
#include <sstream>

#include "TaskMonitor.hpp"
#include "WorkloadStatistics.hpp"


class WorkloadPredictor {

private:
	
	//! Maps a tasktype with its aggregated workload statistics
	typedef std::map< std::string, WorkloadStatistics * > workloads_map_t;
	
	//! Array which contains the number of task instances in each workload
	std::atomic<size_t> _instances[num_workloads];
	
	//! Indexes accumulated and unitary costs through by tasktype
	workloads_map_t _workloads;
	
	//! A spinlock to ensure atomicity within the workloads_map_t
	SpinLock _spinlock;
	
	//! Aggregated execution times of tasks that have completed user code
	std::atomic<size_t> _taskCompletionTimes;
	
	//! The predictor singleton instance
	static WorkloadPredictor *_predictor;
	
	
private:
	
	inline WorkloadPredictor() :
		_workloads(),
		_spinlock(),
		_taskCompletionTimes(0)
	{
		for (unsigned short i = 0; i < num_workloads; ++i) {
			_instances[i] = 0;
		}
	}
	
	
	//! \brief Maps task status identifiers to workload identifiers
	//! \param taskStatus The task status
	//! \return The workload id related to the task status
	inline static workload_t getLoadId(monitoring_task_status_t taskStatus)
	{
		switch (taskStatus) {
			case ready_status:
				return ready_load;
			case pending_status:
			case blocked_status:
				return blocked_load;
			case runtime_status:
			case executing_status:
				return executing_load;
			default:
				return null_workload;
		}
	}
	
	//! \brief Increase the number of instances of a workload
	//! \param loadId The identifier of the workload
	inline void increaseInstances(workload_t loadId)
	{
		++(_instances[loadId]);
	}
	
	//! \brief Decrease the number of instances of a workload
	//! \param loadId The identifier of the workload
	inline void decreaseInstances(workload_t loadId)
	{
		--(_instances[loadId]);
	}
	
	//! \brief Retreive the number of instances of a workload
	//! \param loadId The identifier of the workload
	inline size_t getInstances(workload_t loadId) const
	{
		return _instances[loadId].load();
	}
	
	//! \brief Increase a workload with a task's timing statistics
	//! \param[in] loadId The id of the load to increase
	//! \param[in] label The task's label (type identifier)
	//! \param[in] cost The task's computational cost
	inline void increaseWorkload(workload_t loadId, const std::string &label, size_t cost)
	{
		WorkloadStatistics *statistics = nullptr;
		
		_spinlock.lock();
		
		workloads_map_t::iterator it = _workloads.find(label);
		if (it == _workloads.end()) {
			statistics = new WorkloadStatistics();
			_workloads.emplace(label, statistics);
		}
		else {
			statistics = it->second;
		}
		
		_spinlock.unlock();
		
		assert(statistics != nullptr);
		statistics->increaseAccumulatedCost(loadId, cost);
	}
	
	//! \brief Decrease a workload with a task's timing statistics
	//! \param[in] loadId The id of the load to decrease
	//! \param[in] label The task's label (type identifier)
	//! \param[in] cost The task's computational cost
	inline void decreaseWorkload(workload_t loadId, const std::string &label, size_t cost)
	{
		assert(_workloads.find(label) != _workloads.end());
		
		_spinlock.lock();
		
		// The appropriate map entry is created at task instantiation
		// Substract the computational cost and replace the unitary cost
		_workloads[label]->decreaseAccumulatedCost(loadId, cost);
		
		_spinlock.unlock();
	}
	
	//! \brief Increase the aggregated execution time of tasks that have
	//! completed user code
	//! \param taskCompletionTime The amount of time to increase
	inline void increaseTaskCompletionTimes(size_t taskCompletionTime)
	{
		_taskCompletionTimes += taskCompletionTime;
	}
	
	//! \brief Decrease the aggregated execution time of tasks that have
	//! completed user code
	//! \param taskCompletionTime The amount of time to decrease
	inline void decreaseTaskCompletionTimes(size_t taskCompletionTime)
	{
		_taskCompletionTimes -= taskCompletionTime;
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	WorkloadPredictor(WorkloadPredictor const&) = delete;            // Copy construct
	WorkloadPredictor(WorkloadPredictor&&) = delete;                 // Move construct
	WorkloadPredictor& operator=(WorkloadPredictor const&) = delete; // Copy assign
	WorkloadPredictor& operator=(WorkloadPredictor &&) = delete;     // Move assign
	
	
	//! \brief Initialize workload predictions
	static inline void initialize()
	{
		// Create the predictor singleton
		if (_predictor == nullptr) {
			_predictor = new WorkloadPredictor();
		}
	}
	
	//! \brief Shutdown workload predictions
	static inline void shutdown()
	{
		if (_predictor != nullptr) {
			// Destroy all the Workload statistics
			for (auto const &it : _predictor->_workloads) {
				if (it.second != nullptr) {
					delete it.second;
				}
			}
			
			// Destroy the predictor module
			delete _predictor;
		}
	}
	
	//! \brief Display workload statistics
	//! \param stream The output stream
	static inline void displayStatistics(std::stringstream &stream)
	{
		if (_predictor != nullptr) {
			stream << std::left << std::fixed << std::setprecision(2) << "\n";
			stream << "+-----------------------------+\n";
			stream << "|       WORKLOADS (μs)        |\n";
			stream << "+-----------------------------+\n";
			
			for (unsigned short loadId = 0; loadId < num_workloads; ++loadId) {
				size_t inst = _predictor->getInstances((workload_t) loadId);
				double load = getPredictedWorkload((workload_t) loadId);
				std::string loadDesc = std::string(workloadDescriptions[loadId]) + " (" + std::to_string(inst) + ")";
				
				stream << std::setw(40) << loadDesc << load << " μs\n";
			}
			
			stream << "+-----------------------------+\n\n";
		}
	}
	
	//! \brief Aggregate a task's statistics into workloads
	//! \param taskStatistics The task's statistics
	//! \param taskPredictions The task's predictions
	static void taskCreated(
		TaskStatistics *taskStatistics,
		TaskPredictions *taskPredictions
	);
	
	//! \brief Move the aggregation of a task's statistics between workloads
	//! when the task changes its timing status
	//! \param taskStatistics The task's statistics
	//! \param taskPredictions The task's predictions
	//! \param oldStatus The old timing status
	//! \param newStatus The new timing status
	static void taskChangedStatus(
		TaskStatistics *taskStatistics,
		TaskPredictions *taskPredictions,
		monitoring_task_status_t oldStatus,
		monitoring_task_status_t newStatus
	);
	
	//! \brief Account the task's elapsed time for predictions once it
	//! completes user code
	//! \param taskStatistics The task's statistics
	//! \param taskPredictions The task's predictions
	static void taskCompletedUserCode(
		TaskStatistics *taskStatistics,
		TaskPredictions *taskPredictions
	);
	
	//! \brief Move the aggregation of a task's statistics between workloads
	//! when the task finishes execution
	//! \param taskStatistics The task's statistics
	//! \param taskPredictions The task's predictions
	//! \param oldStatus The old timing status
	static void taskFinished(
		TaskStatistics *taskStatistics,
		TaskPredictions *taskPredictions,
		monitoring_task_status_t oldStatus,
		int ancestorsUpdated
	);
	
	//! \brief Get a timing prediction of a certain workload
	//! \param loadId The workload's id
	static double getPredictedWorkload(workload_t loadId);
	
	//! \brief Retrieve the aggregated execution time of tasks that have
	//! completed user code
	static size_t getTaskCompletionTimes();
	
	//! \brief Retreive the number of instances of a workload
	//! \param loadId The identifier of the workload
	static inline size_t getNumInstances(workload_t loadId)
	{
		assert(_predictor != nullptr);
		
		return _predictor->_instances[loadId].load();
	}
	
};

#endif // WORKLOAD_PREDICTOR_HPP
