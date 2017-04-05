#ifndef INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "../api/InstrumentComputePlaceManagement.hpp"


namespace Instrument {
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		return compute_place_id_t(virtualCPUId);
	}
	
	inline compute_place_id_t createdGPU()
	{
		return compute_place_id_t();
	}
}


#endif // INSTRUMENT_NULL_COMPUTE_PLACE_MANAGEMENT_HPP
