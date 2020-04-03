/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_COMPUTE_PLACE_ID_HPP
#define INSTRUMENT_CTF_COMPUTE_PLACE_ID_HPP

#include <stdint.h>

namespace Instrument {
	class compute_place_id_t {
	public:
		uint16_t _id;

		compute_place_id_t()
			: _id(~0)
		{
		}

		compute_place_id_t(uint16_t id)
			: _id(id)
		{
		}

		bool operator==(compute_place_id_t const &other) const
		{
			return (_id == other._id);
		}
	};
}


#endif // INSTRUMENT_CTF_COMPUTE_PLACE_ID_HPP

