/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GENERIC_IDS_THREAD_ID_HPP
#define INSTRUMENT_GENERIC_IDS_THREAD_ID_HPP


namespace Instrument {
	class thread_id_t {
	public:
		typedef unsigned int inner_type_t;
		
	private:
		inner_type_t _id;
		
	public:
		thread_id_t()
			: _id(~0)
		{
		}
		
		thread_id_t(inner_type_t id)
			: _id(id)
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
		
		bool operator<(thread_id_t other)
		{
			return (_id < other._id);
		}
		
	};
}


#endif // INSTRUMENT_GENERIC_IDS_THREAD_ID_HPP
