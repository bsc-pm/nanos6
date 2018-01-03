/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_COMPUTE_PLACE_ID_HPP
#define INSTRUMENT_COMPUTE_PLACE_ID_HPP


#include <support/ConcurrentUnorderedList.hpp>


namespace Instrument {
	class compute_place_id_t {
	public:
		typedef unsigned int inner_type_t;
		
	private:
		inner_type_t _id;
		ConcurrentUnorderedListSlotManager::Slot _concurrentUnorderedListSlot;
		
	public:
		compute_place_id_t()
			: _id(~0), _concurrentUnorderedListSlot()
		{
		}
		
		compute_place_id_t(inner_type_t id, ConcurrentUnorderedListSlotManager::Slot concurrentUnorderedListSlot)
			: _id(id), _concurrentUnorderedListSlot(concurrentUnorderedListSlot)
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
		
		bool operator<(compute_place_id_t other)
		{
			return (_id < other._id);
		}
		
		ConcurrentUnorderedListSlotManager::Slot getConcurrentUnorderedListSlot() const
		{
			return _concurrentUnorderedListSlot;
		}
		
	};
}


#endif // INSTRUMENT_COMPUTE_PLACE_ID_HPP
