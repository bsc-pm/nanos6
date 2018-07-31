/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_COMPUTE_PLACE_ID_HPP
#define INSTRUMENT_EXTRAE_COMPUTE_PLACE_ID_HPP


namespace Instrument {
	class compute_place_id_t {
	public:
		unsigned int _id;
		unsigned int _NUMANode;
		
		compute_place_id_t()
			: _id(~0), _NUMANode(~0)
		{
		}
		
		compute_place_id_t(unsigned int id, unsigned int NUMANode)
			: _id(id), _NUMANode(NUMANode)
		{
		}
		
		bool operator==(__attribute__((unused)) compute_place_id_t const &other) const
		{
			return (_id == other._id) && (_NUMANode == other._NUMANode);
		}
		
	};
}


#endif // INSTRUMENT_EXTRAE_COMPUTE_PLACE_ID_HPP

