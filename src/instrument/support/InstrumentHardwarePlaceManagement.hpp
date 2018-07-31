/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP


#include <InstrumentComputePlaceId.hpp>

#include <cstddef>


namespace Instrument {
	//! This function is called when the runtime creates a new CPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId, __attribute__((unused)) size_t NUMANode)
	{
		return compute_place_id_t(virtualCPUId);
	}
	
	//! This function is called when the runtime creates a new CUDA GPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	inline compute_place_id_t createdCUDAGPU()
	{
		return compute_place_id_t(-2);
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


#endif // INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP
