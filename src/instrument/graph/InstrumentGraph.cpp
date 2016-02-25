#include "InstrumentGraph.hpp"


namespace Instrument {
	namespace Graph {
		std::atomic<thread_id_t> _nextThreadId(1);
		std::atomic<taskwait_id_t> _nextTaskwaitId(1);
		std::atomic<task_id_t> _nextTaskId(0);
		std::atomic<usermutex_id_t> _nextUsermutexId(0);
		std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId(1);
		
		std::map<WorkerThread *, thread_id_t> _threadToId;
		task_to_info_map_t _taskToInfoMap;
		data_access_map_t _accessIdToAccessMap;
		task_invocation_info_label_map_t _taskInvocationLabel;
		usermutex_to_id_map_t _usermutexToId;
		execution_sequence_t _executionSequence;
		std::map<data_access_id_t, data_access_id_t> _superAccessByAccess;
		std::map<data_access_id_t, data_access_id_t> _firstSubAccessByAccess;
		
		SpinLock _graphLock;
		
		EnvironmentVariable<bool> _showDependencyStructures("NANOS_GRAPH_SHOW_DEPENDENCY_STRUCTURES", false);
		
		
		access_t &getAccess(data_access_id_t dataAccessId)
		{
			return _accessIdToAccessMap[dataAccessId];
		}
		
	}
}
