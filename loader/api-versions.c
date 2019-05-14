/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/


#include <nanos6.h>
#include <nanos6/api-check.h>


nanos6_api_versions_t const __user_code_expected_nanos6_api_versions = {
	.api_check_api_version = nanos6_api_check_api,
	.major_api_version = nanos6_major_api,
	
	.blocking_api_version = nanos6_blocking_api,
	.bootstrap_api_version = nanos6_bootstrap_api,
	.cluster_api_version = nanos6_cluster_api,
	.cuda_device_api_version = nanos6_cuda_device_api,
	.final_api_version = nanos6_final_api,
	.instantiation_api_version = nanos6_instantiation_api,
	.library_mode_api_version = nanos6_library_mode_api,
	.locking_api_version = nanos6_locking_api,
	.monitoring_api_version = nanos6_monitoring_api,
	.polling_api_version = nanos6_polling_api,
	.task_constraints_api_version = nanos6_task_constraints_api,
	.task_execution_api_version = nanos6_task_execution_api,
	.task_info_registration_api_version = nanos6_task_info_registration_api,
	.taskloop_api_version = nanos6_taskloop_api,
	.taskwait_api_version = nanos6_taskwait_api,
};

