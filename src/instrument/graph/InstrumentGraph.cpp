#include "InstrumentGraph.hpp"


namespace Instrument {
	namespace Graph {
		std::atomic<thread_id_t> _nextThreadId(1);
		std::atomic<taskwait_id_t> _nextTaskwaitId(1);
		std::atomic<task_id_t> _nextTaskId(0);
		std::atomic<usermutex_id_t> _nextUsermutexId;
		
		std::map<WorkerThread *, thread_id_t> _threadToId;
		task_to_info_map_t _taskToInfoMap;
		usermutex_to_id_map_t _usermutexToId;
		execution_sequence_t _executionSequence;
		
		SpinLock _graphLock;
	}
}
