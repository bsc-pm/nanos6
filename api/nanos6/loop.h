/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LOOP_H
#define NANOS6_LOOP_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_loop_api
enum nanos6_loop_api_t { nanos6_loop_api = 2 };


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	size_t lower_bound; // Inclusive
	size_t upper_bound; // Exclusive
	size_t grainsize;
	size_t chunksize;
} nanos6_loop_bounds_t;

//! \brief Register the bounds of a taskloop/taskfor
//! 
//! This function registers and initializes the loop bounds of a taskloop/taskfor. This should be called
//! after the creation of the task (see nanos6_create_task) and before the submission of the task
//! (see nanos6_submit_task).
//! 
//! \param[in] task The task handler
//! \param[in] lower_bound The lower bound of the iteration space (inclusive)
//! \param[in] upper_bound The upper bound of the iteration space (exclusive)
//! \param[in] grainsize The minimum number of iterations which should be executed by a task 
//! \param[in] chunksize The minimum number of iterations which should be executed by a chunk
void nanos6_register_loop_bounds(
	void *task,
	size_t lower_bound,
	size_t upper_bound,
	size_t grainsize,
	size_t chunksize
);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_LOOP_H */
