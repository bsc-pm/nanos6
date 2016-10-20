#ifndef INSTRUMENT_GRAPH_LOG_MESSAGE_HPP
#define INSTRUMENT_GRAPH_LOG_MESSAGE_HPP


#include <cassert>
#include <sstream>
#include <string>

#include "../api/InstrumentLogMessage.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskId.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "ExecutionSteps.hpp"


using namespace Instrument::Graph;


namespace Instrument {
	namespace Graph {
		template<typename T>
		inline void fillStream(std::ostringstream &stream, T contents)
		{
			stream << contents;
		}
		
		template<typename T, typename... TS>
		inline void fillStream(std::ostringstream &stream, T content1, TS... contents)
		{
			stream << content1;
			fillStream(stream, contents...);
		}
	}
	
	
	template<typename... TS>
	inline void logMessage(task_id_t triggererTaskId, TS... contents)
	{
		std::ostringstream stream;
		fillStream(stream, contents...);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		
		std::lock_guard<SpinLock> guard(_graphLock);
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		size_t cpuId = 0;
		if (currentThread != nullptr) {
			CPU *cpu = (CPU *) currentThread->getComputePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		log_message_step_t *step = new log_message_step_t(
			cpuId, threadId, triggererTaskId,
			stream.str()
		);
		_executionSequence.push_back(step);
		
	}
}


#endif // INSTRUMENT_GRAPH_LOG_MESSAGE_HPP
