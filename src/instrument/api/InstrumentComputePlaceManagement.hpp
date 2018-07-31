/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP


#include <InstrumentComputePlaceId.hpp>

#include <cstddef>


namespace Instrument {
	//! This function is called when the runtime creates a new CPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	compute_place_id_t createdCPU(unsigned int virtualCPUId, size_t NUMANode);
	
	//! This function is called when the runtime creates a new CUDA GPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	compute_place_id_t createdCUDAGPU();
	
	//! Indicates that a compute place is about to be suspended.
	void suspendingComputePlace(compute_place_id_t const &computePlace);
	
	//! Indicates that a compute place has been resumed.
	void resumedComputePlace(compute_place_id_t const &computePlace);
	
	void shuttingDownComputePlace(compute_place_id_t const &computePlace);
}


#endif // INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP
