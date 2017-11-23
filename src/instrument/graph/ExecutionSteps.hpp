/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_EXECUTION_STEPS_HPP
#define INSTRUMENT_GRAPH_EXECUTION_STEPS_HPP


#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentGraph.hpp"
#include "../generic_ids/InstrumentThreadId.hpp"

#include <sstream>


namespace Instrument {
	namespace Graph {
		struct execution_step_flush_state_t {
			InstrumentationContext _lastContext;
			bool _hasAlreadyFlushed;
			
			execution_step_flush_state_t()
				: _lastContext(), _hasAlreadyFlushed(false)
			{
			}
		};
		
		
		struct execution_step_t {
			InstrumentationContext _instrumentationContext;
			
			execution_step_t(InstrumentationContext const &instrumentationContext)
				: _instrumentationContext(instrumentationContext)
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
				
				bool result = !state._hasAlreadyFlushed && (_instrumentationContext != state._lastContext);
				state._hasAlreadyFlushed = result;
				
				return result;
			}
			
			virtual bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				if (!visible()) {
					return false;
				}
				
				state._hasAlreadyFlushed = false;
				state._lastContext = _instrumentationContext;
				
				return false;
			}
			
			virtual void execute() = 0;
			virtual std::string describe() = 0;
			virtual bool visible() = 0;
			
		protected:
			bool forceFlushAfter(execution_step_flush_state_t &state)
			{
				state._hasAlreadyFlushed = true;
				state._lastContext = _instrumentationContext;
				return true;
			}
			
			inline void emitCPUAndTask(std::ostringstream & oss);
			inline void emitCPU(std::ostringstream & oss);
		};
		
		
		struct create_task_step_t : public execution_step_t {
			task_id_t _newTaskId;
			
			create_task_step_t(InstrumentationContext const &instrumentationContext, task_id_t newTaskId)
				: execution_step_t(instrumentationContext), _newTaskId(newTaskId)
			{
			}
			
			bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				if (!visible()) {
					return false;
				}
				
				state._hasAlreadyFlushed = false;
				state._lastContext = _instrumentationContext;
				state._lastContext._taskId = _newTaskId;
				
				return false;
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct enter_task_step_t : public execution_step_t {
			enter_task_step_t(InstrumentationContext const &instrumentationContext)
				: execution_step_t(instrumentationContext)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct exit_task_step_t : public execution_step_t {
			exit_task_step_t(InstrumentationContext const &instrumentationContext)
				: execution_step_t(instrumentationContext)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct enter_taskwait_step_t : public execution_step_t {
			taskwait_id_t _taskwaitId;
			
			enter_taskwait_step_t(InstrumentationContext const &instrumentationContext, taskwait_id_t taskwaitId)
				: execution_step_t(instrumentationContext), _taskwaitId(taskwaitId)
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
			
			exit_taskwait_step_t(InstrumentationContext const &instrumentationContext, taskwait_id_t taskwaitId)
				: execution_step_t(instrumentationContext), _taskwaitId(taskwaitId)
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
			
			enter_usermutex_step_t(InstrumentationContext const &instrumentationContext, usermutex_id_t usermutexId)
				: execution_step_t(instrumentationContext), _usermutexId(usermutexId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct block_on_usermutex_step_t : public execution_step_t {
			usermutex_id_t _usermutexId;
			
			block_on_usermutex_step_t(InstrumentationContext const &instrumentationContext, usermutex_id_t usermutexId)
				: execution_step_t(instrumentationContext), _usermutexId(usermutexId)
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
			
			exit_usermutex_step_t(InstrumentationContext const &instrumentationContext, usermutex_id_t usermutexId)
				: execution_step_t(instrumentationContext), _usermutexId(usermutexId)
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
			DataAccessRegion _region;
			bool _weak;
			bool _readSatisfied, _writeSatisfied, _globallySatisfied;
			task_id_t _originatorTaskId;
			
			create_data_access_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t superAccessId, data_access_id_t accessId,
				DataAccessType accessType, DataAccessRegion const &region, bool weak,
				bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
				task_id_t originatorTaskId
			)
				: execution_step_t(instrumentationContext),
				_superAccessId(superAccessId), _accessId(accessId),
				_accessType(accessType), _region(region), _weak(weak),
				_readSatisfied(readSatisfied), _writeSatisfied(writeSatisfied), _globallySatisfied(globallySatisfied),
				_originatorTaskId(originatorTaskId)
			{
			}
			
			bool needsFlushAfter(execution_step_flush_state_t &state)
			{
				if (!visible()) {
					return false;
				}
				
				state._hasAlreadyFlushed = false;
				state._lastContext = _instrumentationContext;
				state._lastContext._taskId = _originatorTaskId;
				
				return false;
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId,
				DataAccessType newAccessType, bool newWeakness,
				bool becomesUnsatisfied
			)
				: execution_step_t(instrumentationContext),
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
			bool _globallySatisfied;
			task_id_t _targetTaskId;
			
			data_access_becomes_satisfied_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId,
				bool globallySatisfied,
				task_id_t targetTaskId
			)
				: execution_step_t(instrumentationContext),
				_accessId(accessId),
				_globallySatisfied(globallySatisfied),
				_targetTaskId(targetTaskId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct modified_data_access_region_step_t : public execution_step_t {
			data_access_id_t _accessId;
			DataAccessRegion _region;
			
			modified_data_access_region_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId,
				DataAccessRegion region
			)
				: execution_step_t(instrumentationContext),
				_accessId(accessId),
				_region(region)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct fragment_data_access_step_t : public execution_step_t {
			data_access_id_t _originalAccessId;
			data_access_id_t _newFragmentAccessId;
			DataAccessRegion _newRegion;
			
			fragment_data_access_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t originalAccessId, data_access_id_t newFragmentAccessId,
				DataAccessRegion newRegion
			)
				: execution_step_t(instrumentationContext),
				_originalAccessId(originalAccessId), _newFragmentAccessId(newFragmentAccessId),
				_newRegion(newRegion)
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId, data_access_id_t fragmentAccessId
			)
				: execution_step_t(instrumentationContext),
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t dataAccessId
			)
				: execution_step_t(instrumentationContext),
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId
			)
			: execution_step_t(instrumentationContext),
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId
			)
				: execution_step_t(instrumentationContext),
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
			DataAccessRegion _region;
			bool _direct, _bidirectional;
			bool _producedChanges;
			
			linked_data_accesses_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t sourceAccessId, task_id_t sinkTaskId,
				DataAccessRegion region,
				bool direct, bool bidirectional
			)
				: execution_step_t(instrumentationContext),
				_sourceAccessId(sourceAccessId), _sinkTaskId(sinkTaskId),
				_region(region),
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t sourceAccessId, task_id_t sinkTaskId, bool direct
			)
				: execution_step_t(instrumentationContext),
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
				InstrumentationContext const &instrumentationContext,
				data_access_id_t oldSuperAccessId, data_access_id_t newSuperAccessId,
				data_access_id_t accessId
			)
				: execution_step_t(instrumentationContext),
				_oldSuperAccessId(oldSuperAccessId), _newSuperAccessId(newSuperAccessId),
				_accessId(accessId)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct new_data_access_property_step_t : public execution_step_t {
			data_access_id_t _accessId;
			std::string _shortName;
			std::string _longName;
			
			new_data_access_property_step_t(
				InstrumentationContext const &instrumentationContext,
				data_access_id_t accessId,
				std::string shortName, std::string longName
			)
				: execution_step_t(instrumentationContext),
				_accessId(accessId),
				_shortName(shortName), _longName(longName)
			{
			}
			
			virtual void execute();
			virtual std::string describe();
			virtual bool visible();
		};
		
		
		struct log_message_step_t : public execution_step_t {
			std::string _message;
			
			log_message_step_t(
				InstrumentationContext const &instrumentationContext,
				std::string const &message
			)
				: execution_step_t(instrumentationContext),
				_message(message)
			{
			}
			
			log_message_step_t(
				InstrumentationContext const &instrumentationContext,
				std::string &&message
			)
				: execution_step_t(instrumentationContext),
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
