#ifndef INSTRUMENT_GRAPH_EXECUTION_STEPS_HPP
#define INSTRUMENT_GRAPH_EXECUTION_STEPS_HPP


#include "InstrumentGraph.hpp"
#include "../generic_ids/InstrumentThreadId.hpp"

#include <string>


namespace Instrument {
	namespace Graph {
		struct execution_step_flush_state_t {
			thread_id_t _lastThread;
			task_id_t _lastTask;
			bool _hasAlreadyFlushed;
			
			execution_step_flush_state_t()
				: _lastThread(-1), _lastTask(), _hasAlreadyFlushed(false)
			{
			}
		};
		
		
		struct execution_step_t {
			long _cpu;
			thread_id_t _threadId;
			task_id_t _triggererTaskId;
			
			execution_step_t(long cpu, thread_id_t threadId, task_id_t triggererTaskId)
				: _cpu(cpu), _threadId(threadId), _triggererTaskId(triggererTaskId)
			{
			}
			virtual ~execution_step_t()
			{
			}
			
			bool needsFlushBefore(execution_step_flush_state_t &state)
			{
				if (!visible()) {
					return false;
				}
				
				bool result = !state._hasAlreadyFlushed
					&& ((state._lastThread != _threadId) || (state._lastTask != _triggererTaskId));
				
				state._hasAlreadyFlushed = result;
				
				return result;
			}
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				if (!visible()) {
					return false;
				}
				
				state._hasAlreadyFlushed = false;
				state._lastThread = _threadId;
				state._lastTask = _triggererTaskId;
				return false;
			}
			
			virtual void execute() = 0;
			virtual std::string describe() = 0;
			virtual bool visible() = 0;
			
		protected:
			bool forceFlushAfter(execution_step_flush_state_t &state)
			{
				state._hasAlreadyFlushed = true;
				state._lastThread = _threadId;
				state._lastTask = _triggererTaskId;
				return true;
			}
		};
		
		
		struct create_task_step_t : public execution_step_t {
			task_id_t _parentTaskId;
			
			create_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId, task_id_t parentTaskId)
				: execution_step_t(cpu, threadId, taskId), _parentTaskId(parentTaskId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct enter_task_step_t : public execution_step_t {
			enter_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct exit_task_step_t : public execution_step_t {
			exit_task_step_t(long cpu, thread_id_t threadId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct enter_taskwait_step_t : public execution_step_t {
			taskwait_id_t _taskwaitId;
			
			enter_taskwait_step_t(long cpu, thread_id_t threadId, taskwait_id_t taskwaitId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId), _taskwaitId(taskwaitId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				return forceFlushAfter(state);
			}
		};
		
		
		struct exit_taskwait_step_t : public execution_step_t {
			taskwait_id_t _taskwaitId;
			
			exit_taskwait_step_t(long cpu, thread_id_t threadId, taskwait_id_t taskwaitId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId), _taskwaitId(taskwaitId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				return forceFlushAfter(state);
			}
		};
		
		
		struct enter_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			
			enter_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId), _usermutexId(usermutexId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct block_on_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			
			block_on_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId), _usermutexId(usermutexId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				return forceFlushAfter(state);
			}
		};
		
		
		struct exit_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			
			exit_usermutex_step_t(long cpu, thread_id_t threadId, usermutex_id_t usermutexId, task_id_t taskId)
				: execution_step_t(cpu, threadId, taskId), _usermutexId(usermutexId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				return forceFlushAfter(state);
			}
		};
		
		
		struct create_data_access_step_t : public execution_step_t {
			data_access_id_t _superAccessId;
			data_access_id_t _accessId;
			DataAccessType _accessType;
			DataAccessRange _range;
			bool _weak;
			bool _readSatisfied, _writeSatisfied, _globallySatisfied;
			
			create_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t superAccessId, data_access_id_t accessId,
				DataAccessType accessType, DataAccessRange const &range, bool weak,
				bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
				task_id_t originatorTaskId
			)
				: execution_step_t(cpu, threadId, originatorTaskId),
				_superAccessId(superAccessId), _accessId(accessId),
				_accessType(accessType), _range(range), _weak(weak),
				_readSatisfied(readSatisfied), _writeSatisfied(writeSatisfied), _globallySatisfied(globallySatisfied)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct upgrade_data_access_step_t : public execution_step_t {
			data_access_id_t _accessId;
			DataAccessType _newAccessType;
			bool _newWeakness;
			bool _becomesUnsatisfied;
			
			upgrade_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId,
				DataAccessType newAccessType, bool newWeakness,
				bool becomesUnsatisfied, task_id_t originatorTaskId
			)
				: execution_step_t(cpu, threadId, originatorTaskId),
				_accessId(accessId),
				_newAccessType(newAccessType), _newWeakness(newWeakness),
				_becomesUnsatisfied(becomesUnsatisfied)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct data_access_becomes_satisfied_step_t : public execution_step_t {
			data_access_id_t _accessId;
			bool _readSatisfied, _writeSatisfied, _globallySatisfied;
			task_id_t _targetTaskId;
			
			data_access_becomes_satisfied_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId,
				bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
				task_id_t triggererTaskId, task_id_t targetTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_accessId(accessId),
				_readSatisfied(readSatisfied), _writeSatisfied(writeSatisfied), _globallySatisfied(globallySatisfied),
				_targetTaskId(targetTaskId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct modified_data_access_range_step_t : public execution_step_t {
			data_access_id_t _accessId;
			DataAccessRange _range;
			
			modified_data_access_range_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId,
				DataAccessRange range,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_accessId(accessId),
				_range(range)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct fragment_data_access_step_t : public execution_step_t {
			data_access_id_t _originalAccessId;
			data_access_id_t _newFragmentAccessId;
			DataAccessRange _newRange;
			
			fragment_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t originalAccessId, data_access_id_t newFragmentAccessId,
				DataAccessRange newRange,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_originalAccessId(originalAccessId), _newFragmentAccessId(newFragmentAccessId),
				_newRange(newRange)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct create_subaccess_fragment_step_t : public execution_step_t {
			data_access_id_t _accessId;
			data_access_id_t _fragmentAccessId;
			
			create_subaccess_fragment_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId, data_access_id_t fragmentAccessId,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_accessId(accessId), _fragmentAccessId(fragmentAccessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct completed_data_access_step_t : public execution_step_t {
			data_access_id_t _accessId;
			
			completed_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t dataAccessId,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_accessId(dataAccessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct data_access_becomes_removable_step_t : public execution_step_t {
			data_access_id_t _accessId;
			
			data_access_becomes_removable_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId,
				task_id_t triggererTaskId
			)
			: execution_step_t(cpu, threadId, triggererTaskId),
			_accessId(accessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct removed_data_access_step_t : public execution_step_t {
			data_access_id_t _accessId;
			
			removed_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t accessId,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_accessId(accessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct linked_data_accesses_step_t : public execution_step_t {
			data_access_id_t _sourceAccessId;
			task_id_t _sinkTaskId;
			DataAccessRange _range;
			bool _direct, _bidirectional;
			bool _producedChanges;
			
			linked_data_accesses_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t sourceAccessId, task_id_t sinkTaskId,
				DataAccessRange range,
				bool direct, bool bidirectional,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_sourceAccessId(sourceAccessId), _sinkTaskId(sinkTaskId),
				_range(range),
				_direct(direct), _bidirectional(bidirectional),
				_producedChanges(false)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct unlinked_data_accesses_step_t : public execution_step_t {
			data_access_id_t _sourceAccessId;
			task_id_t _sinkTaskId;
			bool _direct;
			bool _producedChanges;
			
			unlinked_data_accesses_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t sourceAccessId, task_id_t sinkTaskId, bool direct,
				task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_sourceAccessId(sourceAccessId), _sinkTaskId(sinkTaskId), _direct(direct),
				_producedChanges(false)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct reparented_data_access_step_t : public execution_step_t {
			data_access_id_t _oldSuperAccessId;
			data_access_id_t _newSuperAccessId;
			data_access_id_t _accessId;
			
			reparented_data_access_step_t(
				long cpu, thread_id_t threadId,
				data_access_id_t oldSuperAccessId, data_access_id_t newSuperAccessId,
				data_access_id_t accessId, task_id_t triggererTaskId
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_oldSuperAccessId(oldSuperAccessId), _newSuperAccessId(newSuperAccessId),
				_accessId(accessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct log_message_step_t : public execution_step_t {
			std::string _message;
			
			log_message_step_t(
				long cpu, thread_id_t threadId,
				std::string const &message
			)
				: execution_step_t(cpu, threadId, task_id_t()),
				_message(message)
			{
			}
			
			log_message_step_t(
				long cpu, thread_id_t threadId, task_id_t triggererTaskId,
				std::string &&message
			)
				: execution_step_t(cpu, threadId, triggererTaskId),
				_message(std::move(message))
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
	}
}


#endif // INSTRUMENT_GRAPH_EXECUTION_STEPS_HPP
