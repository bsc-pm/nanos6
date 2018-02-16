/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_ID_HPP
#define INSTRUMENT_STATS_TASK_ID_HPP


namespace Instrument {
	namespace Stats {
		struct TaskTypeAndTimes;
	}
	
	//! This is the default task identifier for the instrumentation.
	struct task_id_t {
		typedef Stats::TaskTypeAndTimes *contents_t;
		
		contents_t _contents;
		
		task_id_t()
			:_contents(nullptr)
		{
		}
		
		task_id_t(Stats::TaskTypeAndTimes *other)
			: _contents(other)
		{
		}
		
		operator contents_t()
		{
			return _contents;
		}
		
		contents_t operator->() const
		{
			return _contents;
		}
		contents_t operator->()
		{
			return _contents;
		}
		
		Stats::TaskTypeAndTimes const &operator*() const
		{
			return *_contents;
		}
		Stats::TaskTypeAndTimes &operator*()
		{
			return *_contents;
		}
		
		bool operator==(task_id_t const &other) const
		{
			return (_contents == other._contents);
		}
		
		bool operator!=(task_id_t const &other) const
		{
			return (_contents != other._contents);
		}
		
		operator long() const
		{
			return -1;
		}
	};
}

#endif // INSTRUMENT_STATS_TASK_ID_HPP
