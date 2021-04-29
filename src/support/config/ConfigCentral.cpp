/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/


#include <config.h>

#include "ConfigCentral.hpp"
#include "hardware/HardwareInfo.hpp"


ConfigCentral::ConfigCentral() :
	_descriptors(),
	_defaults(),
	_listDefaults()
{
	// Cluster
	registerOption<string_t>("cluster.communication", "disabled");
	registerOption<memory_t>("cluster.distributed_memory", 2UL << 30);
	registerOption<memory_t>("cluster.local_memory", 0);
	registerOption<string_t>("cluster.scheduling_policy", "locality");
	registerOption<integer_t>("cluster.va_start", 0);

	// CPU manager
	registerOption<string_t>("cpumanager.policy", "default");

	// CUDA devices
	registerOption<integer_t>("devices.cuda.page_size", 0x8000);
	registerOption<integer_t>("devices.cuda.streams", 16);
	registerOption<bool_t>("devices.cuda.polling.pinned", true);
	registerOption<integer_t>("devices.cuda.polling.period_us", 1000);

	// OpenACC devices
	registerOption<integer_t>("devices.openacc.default_queues", 64);
	registerOption<integer_t>("devices.openacc.max_queues", 128);
	registerOption<bool_t>("devices.openacc.polling.pinned", true);
	registerOption<integer_t>("devices.openacc.polling.period_us", 1000);

	// DLB
	registerOption<bool_t>("dlb.enabled", false);

	// Hardware counters
	registerOption<bool_t>("hardware_counters.verbose", false);
	registerOption<string_t>("hardware_counters.verbose_file", "nanos6-output-hwcounters.txt");

	// RAPL hardware counters
	registerOption<bool_t>("hardware_counters.rapl.enabled", false);

	// PAPI hardware counters
	registerOption<bool_t>("hardware_counters.papi.enabled", false);
	registerOption<string_t>("hardware_counters.papi.counters", {});

	// PQOS hardware counters
	registerOption<bool_t>("hardware_counters.pqos.enabled", false);
	registerOption<string_t>("hardware_counters.pqos.counters", {});

	// CTF instrumentation
	registerOption<bool_t>("instrument.ctf.converter.enabled", true);
	registerOption<string_t>("instrument.ctf.converter.location", "");
	registerOption<string_t>("instrument.ctf.events.kernel.exclude", {});
	registerOption<string_t>("instrument.ctf.events.kernel.file", "");
	registerOption<string_t>("instrument.ctf.events.kernel.presets", {});
	registerOption<string_t>("instrument.ctf.tmpdir", "");

	// Extrae instrumentation
	registerOption<bool_t>("instrument.extrae.as_threads", false);
	registerOption<integer_t>("instrument.extrae.detail_level", 1);

	// Graph instrumentation
	registerOption<bool_t>("instrument.graph.display", false);
	registerOption<string_t>("instrument.graph.display_command", "xdg-open");
	registerOption<bool_t>("instrument.graph.shorten_filenames", false);
	registerOption<bool_t>("instrument.graph.show_all_steps", false);
	registerOption<bool_t>("instrument.graph.show_dead_dependency_structures", false);
	registerOption<bool_t>("instrument.graph.show_dead_dependencies", false);
	registerOption<bool_t>("instrument.graph.show_dependency_structures", false);
	registerOption<bool_t>("instrument.graph.show_log", false);
	registerOption<bool_t>("instrument.graph.show_regions", false);
	registerOption<bool_t>("instrument.graph.show_spurious_dependency_structures", false);
	registerOption<bool_t>("instrument.graph.show_superaccess_links", true);

	// Stats instrumentation
	registerOption<string_t>("instrument.stats.output_file", "/dev/stderr");

	// Verbose instrumentation
	registerOption<string_t>("instrument.verbose.areas", {
		"all", "!ComputePlaceManagement", "!DependenciesByAccess",
		"!DependenciesByAccessLinks", "!DependenciesByGroup",
		"!LeaderThread", "!TaskStatus", "!ThreadManagement"
	});
	registerOption<bool_t>("instrument.verbose.dump_only_on_exit", false);
	registerOption<string_t>("instrument.verbose.output_file", "/dev/stderr");
	registerOption<bool_t>("instrument.verbose.timestamps", true);

	// Loader
	registerOption<bool_t>("loader.verbose", false);
	registerOption<bool_t>("loader.warn_envars", true);
	registerOption<string_t>("loader.library_path", "");
	registerOption<string_t>("loader.report_prefix", "");

	// Memory allocator
	registerOption<memory_t>("memory.pool.global_alloc_size", 8 * 1024 * 1024);
	registerOption<memory_t>("memory.pool.chunk_size", 128 * 1024);

	// Miscellaneous
	registerOption<memory_t>("misc.stack_size", 8 * 1024 * 1024);

	// Monitoring
	registerOption<integer_t>("monitoring.cpuusage_prediction_rate", 100);
	registerOption<bool_t>("monitoring.enabled", false);
	registerOption<integer_t>("monitoring.rolling_window", 20);
	registerOption<bool_t>("monitoring.verbose", true);
	registerOption<string_t>("monitoring.verbose_file", "output-monitoring.txt");
	registerOption<bool_t>("monitoring.wisdom", false);

	// NUMA support
	registerOption<bool_t>("numa.discover", true);
	registerOption<bool_t>("numa.report", false);
	registerOption<bool_t>("numa.scheduling", true);
	registerOption<string_t>("numa.tracking", "auto");

	// Scheduler
	registerOption<bool_t>("scheduler.immediate_successor", true);
	registerOption<string_t>("scheduler.policy", "fifo");
	registerOption<bool_t>("scheduler.priority", true);

	// Taskfor
	registerOption<integer_t>("taskfor.groups", 1);
	registerOption<bool_t>("taskfor.report", false);

	// Throttle
	registerOption<bool_t>("throttle.enabled", false);
	registerOption<memory_t>("throttle.max_memory", 0);
	registerOption<integer_t>("throttle.polling_period_us", 1000);
	registerOption<integer_t>("throttle.pressure", 70);
	registerOption<integer_t>("throttle.tasks", 5000000);

	// Turbo
	registerOption<bool_t>("turbo.enabled", false);

	// Version details
	registerOption<bool_t>("version.debug", false);
	registerOption<string_t>("version.dependencies", "discrete");
	registerOption<string_t>("version.instrument", "none");
}
