/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_TYPE_DATA_HPP
#define TASK_TYPE_DATA_HPP

// NOTE: "array_wrapper" must be included before any other boost modules
// Workaround for a missing include in boost 1.64
#include <boost/version.hpp>
#if BOOST_VERSION == 106400
#include <boost/serialization/array_wrapper.hpp>
#endif

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "lowlevel/SpinLock.hpp"

namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;


//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TaskTypeData {

private:

	//! A spinlock so that tasktype structures can be accessed atomically

	/*    HARDWARE COUNTERS    */

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::sum, BoostAccTag::mean, BoostAccTag::variance, BoostAccTag::count> > statistics_accumulator_t;
	typedef std::vector<statistics_accumulator_t> counter_statistics_t;

	//! A vector of hardware counter accumulators
	counter_statistics_t _counterStatistics;

	//! A spinlock to access hardware counter structures
	SpinLock _counterLock;

	/*    INSTRUMENTATION    */

	/*    MONITORING    */

public:

	inline TaskTypeData() :
		_counterStatistics(HWCounters::TOTAL_NUM_EVENTS)
	{
	}

	/*    HARDWARE COUNTERS    */

	//! \brief Accumulate the counters of a task for statistics purposes
	//!
	//! \param[in] counterType The type of counter
	//! \param[in] counterValue The accumulated value of the counter for a task
	inline void addCounter(HWCounters::counters_t counterType, double counterValue)
	{
		assert(counterType < _counterStatistics.size());

		_counterLock.lock();
		_counterStatistics[counterType](counterValue);
		_counterLock.unlock();
	}

	//! \brief Retreive, for a certain type of counter, the sum of accumulated
	//! values of all tasks from this type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the sum of accumulated values
	inline double getCounterSum(HWCounters::counters_t counterType)
	{
		return BoostAcc::sum(_counterStatistics[counterType]);
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated values of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the avaerage accumulated value
	inline double getCounterAvg(HWCounters::counters_t counterType)
	{
		return BoostAcc::mean(_counterStatistics[counterType]);
	}

	//! \brief Retreive, for a certain type of counter, the standard deviation
	//! taking into account all the values in the accumulator of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the standard deviation of the counter
	inline double getCounterStddev(HWCounters::counters_t counterType)
	{
		return sqrt(BoostAcc::variance(_counterStatistics[counterType]));
	}

	//! \brief Retreive, for a certain type of counter, the amount of values
	//! in the accumulator (i.e., the number of tasks)
	//!
	//! \param[in] counterType The type of counter
	//! \return A size_t with the number of accumulated values
	inline size_t getCounterCount(HWCounters::counters_t counterType)
	{
		return BoostAcc::count(_counterStatistics[counterType]);
	}

	/*    INSTRUMENTATION    */

	/*    MONITORING    */

};

#endif // TASK_TYPE_DATA_HPP
