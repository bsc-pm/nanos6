#ifndef INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP


#include <InstrumentComputePlaceId.hpp>


namespace Instrument {
	//! This function is called when the runtime creates a new CPU and
	//! must return an instrumentation-specific computePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId)
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
}


#endif // INSTRUMENT_SUPPORT_COMPUTE_PLACE_MANAGEMENT_HPP
