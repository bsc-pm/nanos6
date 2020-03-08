/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_ID_HPP
#define INSTRUMENT_TASK_ID_HPP


namespace Instrument {
	enum state_t
	{
		INITIAL = 0,
		STARTED,
		FINISHED
	};
	
	class task_id_t {
	public:
		typedef int inner_type_t;
		
	private:
		inner_type_t _id;
		state_t _state;
	
	public:
		task_id_t()
			: _id(-1), _state(INITIAL)
		{
		}
		
		task_id_t(inner_type_t id)
			: _id(id), _state(INITIAL)
		{
		}
		
		operator inner_type_t() const
		{
			return _id;
		}
		
		bool operator==(inner_type_t other) const
		{
			return (_id == other);
		}
		
		bool operator!=(inner_type_t other) const
		{
			return (_id != other);
		}
		
		bool operator<(task_id_t other)
		{
			return (_id < other._id);
		}

		state_t getState()
		{
			return _state;
		}
		
		void setState(state_t state)
		{
			_state = state;
		}
		
	};
	
}

#endif // INSTRUMENT_TASK_ID_HPP
