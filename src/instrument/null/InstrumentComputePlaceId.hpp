/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_COMPUTE_PLACE_ID_HPP
#define INSTRUMENT_NULL_COMPUTE_PLACE_ID_HPP


namespace Instrument {
	class compute_place_id_t {
	public:
		compute_place_id_t()
		{
		}
		
		template<typename T>
		compute_place_id_t(__attribute__((unused)) T id)
		{
		}
		
		bool operator==(__attribute__((unused)) compute_place_id_t const &other) const
		{
			return true;
		}
		
	};
}


#endif // INSTRUMENT_NULL_COMPUTE_PLACE_ID_HPP

