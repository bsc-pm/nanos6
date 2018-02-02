/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_TASK_ID_HPP
#define INSTRUMENT_EXTRAE_TASK_ID_HPP


#include <nanos6.h>

#include <lowlevel/SpinLock.hpp>

#include "InstrumentExtrae.hpp"

#include <atomic>
#include <set>


namespace Instrument {
	namespace Extrae {
		typedef std::pair<size_t, dependency_tag_t> predecessor_entry_t; // Task and strength
		
		struct TaskInfo;
	}
	
	
	struct task_id_t {
		Extrae::TaskInfo *_taskInfo;
		
		task_id_t()
			: _taskInfo(nullptr)
		{
		}
		
		task_id_t(Extrae::TaskInfo *taskInfo)
			: _taskInfo(taskInfo)
		{
		}
		
		task_id_t(task_id_t const &other)
			: _taskInfo(other._taskInfo)
		{
		}
		
		
		bool operator==(task_id_t const &other) const
		{
			return (_taskInfo == other._taskInfo);
		}
		
		bool operator!=(task_id_t const &other) const
		{
			return (_taskInfo != other._taskInfo);
		}
		
		bool operator<(task_id_t const &other) const
		{
			return (_taskInfo < other._taskInfo);
		}
		
	};
	
	
	namespace Extrae {
		struct TaskInfo {
			nanos_task_info *_taskInfo;
			size_t _taskId;
			int _nestingLevel;
			Instrument::Extrae::TaskInfo *_parent;
			
			std::atomic<bool> _inTaskwait;
			
			SpinLock _lock;
			std::set<predecessor_entry_t> _predecessors;
			
			TaskInfo()
				: _taskInfo(nullptr), _taskId(~0UL), _nestingLevel(-1), _parent(),
				_inTaskwait(false), _lock(), _predecessors()
			{
			}
			
			TaskInfo(nanos_task_info *taskInfo, int nestingLevel, Instrument::Extrae::TaskInfo *parent)
				: _taskInfo(taskInfo), _nestingLevel(nestingLevel), _parent(parent),
				_inTaskwait(false), _lock(), _predecessors()
			{
				_taskId = _nextTaskId++;
			}
		};
	}
}


#endif // INSTRUMENT_EXTRAE_TASK_ID_HPP
