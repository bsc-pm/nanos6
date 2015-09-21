#ifndef INSTRUMENT_GRAPH_GRAPH_HPP
#define INSTRUMENT_GRAPH_GRAPH_HPP


#include "api/nanos6_rt_interface.h"

#include "lowlevel/SpinLock.hpp"
#include "system/ompss/UserMutex.hpp"

#include <InstrumentTaskId.hpp>

#include <atomic>
#include <list>
#include <map>
#include <deque>


class WorkerThread;


namespace Instrument {
	
	namespace Graph {
		typedef long taskwait_id_t;
		typedef long usermutex_id_t;
		typedef long thread_id_t;
		
		enum task_status_t {
			not_created_status,
			not_started_status,
			started_status,
			blocked_status,
			finished_status
		};
		
		//! \brief this is the list of direct children between the previous (if any) and next (if any) taskwait
		typedef std::deque<task_id_t> children_list_t;
		
		struct phase_t {
			virtual ~phase_t()
			{
			}
			
		};
		
		struct task_group_t : public phase_t {
			children_list_t _children;
			taskwait_id_t _clusterId;
			
			task_group_t(taskwait_id_t id)
				: phase_t(), _clusterId(id)
			{
			}
		};
		
		struct taskwait_t : public phase_t {
			taskwait_id_t _taskwaitId;
			char const *_taskwaitSource;
			
			taskwait_t(taskwait_id_t taskwaitId, char const *taskwaitSource)
				: _taskwaitId(taskwaitId), _taskwaitSource(taskwaitSource)
			{
			}
		};
		
		typedef std::deque<phase_t *> phase_list_t;
		
		struct task_info_t {
			nanos_task_info *_nanos_task_info;
			nanos_task_invocation_info *_nanos_task_invocation_info;
			
			task_status_t _status;
			long _lastCPU;
			
			phase_list_t _phaseList;
			
			task_info_t()
				: _nanos_task_info(nullptr), _nanos_task_invocation_info(nullptr),
				_status(not_created_status), _lastCPU(-1),
				_phaseList()
			{
			}
		};
		
		//! \brief maps tasks to their information
		typedef std::map<task_id_t, task_info_t> task_to_info_map_t;
		
		//! \brief maps task invocations to the text to use as label
		typedef std::map<nanos_task_invocation_info *, std::string> task_invocation_info_label_map_t;
		
		
		struct execution_step_t {
			long _cpu;
			thread_id_t _threadId;
			
			execution_step_t(long cpu, thread_id_t threadId)
				: _cpu(cpu), _threadId(threadId)
			{
			}
			virtual ~execution_step_t()
			{
			}
		};
		
		struct create_task_step_t : public execution_step_t {
			task_id_t _taskId;
			task_id_t _parentTaskId;
			
			create_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId, task_id_t parentTaskId)
			: execution_step_t(cpu, threadId), _taskId(taskId), _parentTaskId(parentTaskId)
			{
			}
		};
		
		struct enter_task_step_t : public execution_step_t {
			task_id_t _taskId;
			
			enter_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId)
				: execution_step_t(cpu, threadId), _taskId(taskId)
			{
			}
		};
		
		struct exit_task_step_t : public execution_step_t {
			task_id_t _taskId;
			
			exit_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId)
				: execution_step_t(cpu, threadId), _taskId(taskId)
			{
			}
		};
		
		struct enter_taskwait_step_t : public execution_step_t {
			taskwait_id_t _taskwaitId;
			task_id_t _taskId;
			
			enter_taskwait_step_t(long cpu, thread_id_t threadId, taskwait_id_t taskwaitId, task_id_t taskId)
				: execution_step_t(cpu, threadId), _taskwaitId(taskwaitId), _taskId(taskId)
			{
			}
		};
		
		struct exit_taskwait_step_t : public execution_step_t {
			taskwait_id_t _taskwaitId;
			task_id_t _taskId;
			
			exit_taskwait_step_t(long cpu, thread_id_t threadId, taskwait_id_t taskwaitId, task_id_t taskId)
				: execution_step_t(cpu, threadId), _taskwaitId(taskwaitId), _taskId(taskId)
			{
			}
		};
		
		struct enter_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			task_id_t _taskId;
			
			enter_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
			: execution_step_t(cpu, threadId), _usermutexId(usermutexId), _taskId(taskId)
			{
			}
		};
		
		struct block_on_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			task_id_t _taskId;
			
			block_on_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
			: execution_step_t(cpu, threadId), _usermutexId(usermutexId), _taskId(taskId)
			{
			}
		};
		
		struct exit_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			task_id_t _taskId;
			
			exit_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
			: execution_step_t(cpu, threadId), _usermutexId(usermutexId), _taskId(taskId)
			{
			}
		};
		
		typedef std::list<execution_step_t *> execution_sequence_t;
		
		
		extern std::atomic<thread_id_t> _nextThreadId;
		extern std::atomic<taskwait_id_t> _nextTaskwaitId;
		extern std::atomic<task_id_t> _nextTaskId;
		extern std::atomic<usermutex_id_t> _nextUsermutexId;
		
		
		//! \brief maps thread pointers to thread identifiers
		extern std::map<WorkerThread *, thread_id_t> _threadToId;
		
		//! \brief maps task identifiers to their information
		extern task_to_info_map_t _taskToInfoMap;
		
		//! \brief maps task invocation struct addresses to the text to use as task label
		extern task_invocation_info_label_map_t _taskInvocationLabel;
		
		typedef std::map<UserMutex *, usermutex_id_t> usermutex_to_id_map_t;
		
		//! \brief maps user mutexes to a numeric identifier
		extern usermutex_to_id_map_t _usermutexToId;
		
		//! \brief sequence of task executions with their corresponding CPU
		extern execution_sequence_t _executionSequence;
		
		extern SpinLock _graphLock;
	};
	
}


#endif // INSTRUMENT_GRAPH_GRAPH_HPP
