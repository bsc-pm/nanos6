/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_INSTRUMENT_H
#define NANOS6_INSTRUMENT_H

#include <stdint.h>

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_instrument_api
enum nanos6_instrument_api_t { nanos6_instrument_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	//! \brief Clock offset (ns) of the current process
	int64_t offset_ns;

	//! \brief Number of samples used to compute the offset stdev
	uint64_t num_samples;

	//! \brief Standard deviation of clock offset (ns) in a sample of size num_samples
	double stdev_ns;
} nanos6_clock_offset_t;

typedef struct
{
	//! \brief The rank of the current process in the distributed execution
	uint64_t rank;

	//! \brief The number of ranks in the distributed execution
	uint64_t num_ranks;

	//! \brief The local clock offset respecting the reference clock computed across all ranks
	nanos6_clock_offset_t clock_offset;
} nanos6_distributed_instrument_info_t;

//! \brief Check whether the instrumentation in distributed executions is enabled
int nanos6_is_distributed_instrument_enabled(void);

//! \brief Setup the instrumentation in distributed executions
//!
//! This API provides nanos6 with distributed memory information for the current
//! process. It might be called from within a task (typically the main task).
//! A distributed memory barrier (such as MPI_Barrier()) must be called after this API.
//!
//! \param[in] info The information for distributed instrumentation
void nanos6_setup_distributed_instrument(const nanos6_distributed_instrument_info_t *info);

//! \brief Get the start time of the instrumentation (ns)
int64_t nanos6_get_instrument_start_time_ns(void);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif // NANOS6_INSTRUMENT_H
