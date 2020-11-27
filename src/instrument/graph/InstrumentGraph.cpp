/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentGraph.hpp"

#include <InstrumentTaskId.hpp>


namespace Instrument {
	namespace Graph {
		std::map<taskwait_id_t, taskwait_t *> _taskwaitToInfoMap;

		std::atomic<taskwait_id_t::inner_type_t> _nextTaskwaitId(1);
		std::atomic<task_id_t::inner_type_t> _nextTaskId(0);
		std::atomic<usermutex_id_t> _nextUsermutexId(0);
		std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId(1);

		task_to_info_map_t _taskToInfoMap;
		data_access_map_t _accessIdToAccessMap;
		task_invocation_info_label_map_t _taskInvocationLabel;
		usermutex_to_id_map_t _usermutexToId;
		execution_sequence_t _executionSequence;

		SpinLock _graphLock;

		ConfigVariable<bool> _showDependencyStructures("instrument.graph.show_dependency_structures");
		ConfigVariable<bool> _showRegions("instrument.graph.show_regions");
		ConfigVariable<bool> _showLog("instrument.graph.show_log");
	}
}
