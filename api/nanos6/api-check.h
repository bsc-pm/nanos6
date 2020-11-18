/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_API_CHECK_H
#define NANOS6_API_CHECK_H

#if HAVE_CONFIG_H
#include "config.h"
#endif

#include "blocking.h"
#include "bootstrap.h"
#include "cluster.h"
#include "final.h"
#include "library-mode.h"
#include "lint.h"
#include "loop.h"
#include "major.h"
#include "monitoring.h"
#include "polling.h"
#include "reductions.h"
#include "task-info-registration.h"
#include "task-instantiation.h"
#include "taskwait.h"
#include "user-mutex.h"


#if USE_CUDA
#include "cuda_device.h"
#else
enum nanos6_cuda_device_api_t { nanos6_cuda_device_api = 1 };
#endif

#if USE_OPENACC
#include "openacc_device.h"
#else
enum nanos6_openacc_device_api_t { nanos6_openacc_device_api = 1 };
#endif

#pragma GCC visibility push(default)

enum nanos6_api_check_api_t { nanos6_api_check_api = 7 };


#ifdef __cplusplus
extern "C" {
#endif


// NOTE: the value of nanos6_api_check_api needs to be updated whenever this struct is updated
typedef struct {
	enum nanos6_api_check_api_t api_check_api_version;
	enum nanos6_major_api_t major_api_version;

	enum nanos6_blocking_api_t blocking_api_version;
	enum nanos6_bootstrap_api_t bootstrap_api_version;
	enum nanos6_cluster_api_t cluster_api_version;
	enum nanos6_config_api_t config_api_version;
	enum nanos6_cuda_device_api_t cuda_device_api_version;
	enum nanos6_openacc_device_api_t openacc_device_api_version;
	enum nanos6_final_api_t final_api_version;
	enum nanos6_instantiation_api_t instantiation_api_version;
	enum nanos6_library_mode_api_t library_mode_api_version;
	enum nanos6_lint_api_t lint_api_version;
	enum nanos6_locking_api_t locking_api_version;
	enum nanos6_monitoring_api_t monitoring_api_version;
	enum nanos6_polling_api_t polling_api_version;
	enum nanos6_reductions_api_t reductions_api_version;
	enum nanos6_task_constraints_api_t task_constraints_api_version;
	enum nanos6_task_execution_api_t task_execution_api_version;
	enum nanos6_task_info_registration_api_t task_info_registration_api_version;
	enum nanos6_loop_api_t loop_api_version;
	enum nanos6_taskwait_api_t taskwait_api_version;
} nanos6_api_versions_t;


//! \brief checks if the runtime API is the one that is expected
//!
//! \returns 1 if the API matches, otherwise 0
int nanos6_check_api_versions(nanos6_api_versions_t const *api_versions);


#ifdef __cplusplus
}
#endif


#pragma GCC visibility pop

#endif /* NANOS6_API_CHECK_H */
