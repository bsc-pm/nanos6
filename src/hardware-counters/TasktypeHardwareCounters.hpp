/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_HARDWARE_COUNTERS_HPP
#define TASKTYPE_HARDWARE_COUNTERS_HPP

// NOTE: "array_wrapper" must be included before any other boost modules
// Workaround for a missing include in boost 1.64
#include <boost/version.hpp>
#if BOOST_VERSION == 106400
#include <boost/serialization/array_wrapper.hpp>
#endif

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "SupportedHardwareCounters.hpp"
#include "lowlevel/SpinLock.hpp"

namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;


//! \brief Use to hold hardware counters on a per-tasktype basis
class TasktypeHardwareCounters {

private:

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::sum, BoostAccTag::mean, BoostAccTag::variance, BoostAccTag::count> > statistics_accumulator_t;
	typedef std::vector<statistics_accumulator_t> counter_statistics_t;

	//! A vector of hardware counter accumulators
	counter_statistics_t _counterStatistics;

	//! A spinlock to access hardware counter structures
	SpinLock _counterLock;

public:

	inline TasktypeHardwareCounters() :
		_counterStatistics(HWCounters::TOTAL_NUM_EVENTS)
	{
	}

	//! \brief Accumulate the counters of a task for statistics purposes
	//!
	//! \param[in] countersToAdd A vector of counter identifiers and values
	inline void addCounters(const std::vector<std::pair<HWCounters::counters_t, double>> &countersToAdd)
	{
		_counterLock.lock();
		for (size_t i = 0; i < countersToAdd.size(); ++i) {
			assert(countersToAdd[i].first < _counterStatistics.size());
			_counterStatistics[countersToAdd[i].first](countersToAdd[i].second);
		}
		_counterLock.unlock();
	}

	//! \brief Retreive, for a certain type of counter, the sum of accumulated
	//! values of all tasks from this type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the sum of accumulated values
	inline double getCounterSum(HWCounters::counters_t counterType)
	{
		_counterLock.lock();
		double sum = BoostAcc::sum(_counterStatistics[counterType]);
		_counterLock.unlock();

		return sum;
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated values of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the average accumulated value
	inline double getCounterAvg(HWCounters::counters_t counterType)
	{
		_counterLock.lock();
		double avg = BoostAcc::mean(_counterStatistics[counterType]);
		_counterLock.unlock();

		return avg;
	}

	//! \brief Retreive, for a certain type of counter, the standard deviation
	//! taking into account all the values in the accumulator of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the standard deviation of the counter
	inline double getCounterStddev(HWCounters::counters_t counterType)
	{
		_counterLock.lock();
		double stddev = sqrt(BoostAcc::variance(_counterStatistics[counterType]));
		_counterLock.unlock();

		return stddev;
	}

	//! \brief Retreive, for a certain type of counter, the amount of values
	//! in the accumulator (i.e., the number of tasks)
	//!
	//! \param[in] counterType The type of counter
	//! \return A size_t with the number of accumulated values
	inline size_t getCounterCount(HWCounters::counters_t counterType)
	{
		_counterLock.lock();
		size_t count = BoostAcc::count(_counterStatistics[counterType]);
		_counterLock.unlock();

		return count;
	}

};

#endif // TASKTYPE_HARDWARE_COUNTERS_HPP
