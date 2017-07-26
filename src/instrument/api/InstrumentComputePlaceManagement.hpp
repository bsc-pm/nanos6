/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP


#include <InstrumentComputePlaceId.hpp>


namespace Instrument {
	//! This function is called when the runtime creates a new CPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	compute_place_id_t createdCPU(unsigned int virtualCPUId);
	
	//! This function is called when the runtime creates a new CUDA GPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	compute_place_id_t createdCUDAGPU();
}


#endif // INSTRUMENT_COMPUTE_PLACE_MANAGEMENT_HPP
