/*
	This file is part of ALPI and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2025 Barcelona Supercomputing Center (BSC)
*/

#ifndef ALPI_DEFS_H
#define ALPI_DEFS_H

/**
 * List of optional API features of ALPI
 * | ----------------------- | -------------------------- |
 * | ALPI_FEATURE_BLOCKING   | alpi_task_block            |
 * | 0x1                     | alpi_task_unblock          |
 * |                         | alpi_task_waitfor_ns       |
 * | ----------------------- | -------------------------- |
 * | ALPI_FEATURE_EVENTS     | alpi_task_events_increase  |
 * | 0x2                     | alpi_task_events_test      |
 * |                         | alpi_task_events_decrease  |
 * | ----------------------- | -------------------------- |
 * | ALPI_FEATURE_RESOURCES  | alpi_cpu_count             |
 * | 0x4                     | alpi_cpu_logical_id        |
 * |                         | alpi_cpu_system_id         |
 * | ----------------------- | -------------------------- |
 * | ALPI_FEATURE_SUSPEND    | alpi_task_suspend_mode_set |
 * | 0x8                     | alpi_task_suspend          |
 * | ----------------------- | -------------------------- |
 */
#define ALPI_FEATURE_BLOCKING  (1 << 0) //! Blocking API checking
#define ALPI_FEATURE_EVENTS    (1 << 1) //! Task events API
#define ALPI_FEATURE_RESOURCES (1 << 2) //! Resource checking

/**
 * @brief Bitmask representing the supported ALPI features
 *
 * This is the combination of individual feature bitmask flag definitions using
 * a bitwise OR. It indicates which features are supported by the implementor
 */
#define ALPI_SUPPORTED_FEATURES ( \
	ALPI_FEATURE_BLOCKING  | \
	ALPI_FEATURE_EVENTS    | \
	ALPI_FEATURE_RESOURCES   \
)

#endif // ALPI_DEFS_H
