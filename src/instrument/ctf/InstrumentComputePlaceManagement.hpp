/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_CTF_COMPUTE_PLACE_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "../api/InstrumentComputePlaceManagement.hpp"
#include <CTFAPI.hpp>


namespace Instrument {

	inline compute_place_id_t createdGPU()
	{
		return compute_place_id_t();
	}

	inline void suspendingComputePlace(compute_place_id_t const &computePlace)
	{
		CTFAPI::tp_cpu_idle(computePlace._id);
	}

	inline void resumedComputePlace(compute_place_id_t const &computePlace)
	{
		// TODO this should be triggered when the CPU's workers wakes
		// up. Not when we ask it to wake up. We need another kind of
		// instrumentation point here.
		CTFAPI::tp_cpu_resume(computePlace._id);
	}

	inline void shuttingDownComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
	}
}


#endif // INSTRUMENT_CTF_COMPUTE_PLACE_MANAGEMENT_HPP
