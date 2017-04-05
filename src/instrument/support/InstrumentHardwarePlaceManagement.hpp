#ifndef INSTRUMENT_SUPPORT_HARDWARE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_SUPPORT_HARDWARE_PLACE_MANAGEMENT_HPP


#include <InstrumentHardwarePlaceId.hpp>


namespace Instrument {
	//! This function is called when the runtime creates a new CPU and
	//! must return an instrumentation-specific hardwarePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	inline hardware_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		return hardware_place_id_t(virtualCPUId);
	}
	
	//! This function is called when the runtime creates a new CUDA GPU and
	//! must return an instrumentation-specific hardwarePlace identifier that will
	//! be used to identify it throughout the rest of the instrumentation API.
	inline hardware_place_id_t createdCUDAGPU()
	{
		return hardware_place_id_t(-2);
	}
}


#endif // INSTRUMENT_SUPPORT_HARDWARE_PLACE_MANAGEMENT_HPP
