/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_TASK_INFO_REGISTRATION_H
#define NANOS6_TASK_INFO_REGISTRATION_H

#include "major.h"

#include "task-instantiation.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_task_info_registration_api
enum nanos6_task_info_registration_api_t { nanos6_task_info_registration_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Register a type of task
//! 
//! \param[in] task_info a pointer to the nanos6_task_info_t structure
void nanos6_register_task_info(nanos6_task_info_t *task_info);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_TASK_INFO_REGISTRATION_H */
