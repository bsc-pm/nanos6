/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "../api/InstrumentComputePlaceManagement.hpp"


namespace Instrument {
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId, __attribute__((unused)) size_t NUMANode)
	{
		return compute_place_id_t(virtualCPUId);
	}
	
	inline compute_place_id_t createdGPU()
	{
		return compute_place_id_t();
	}
	
	inline void suspendingComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
	}
	
	inline void resumedComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
	}
	
	inline void shuttingDownComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
	}
}


#endif // INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP
