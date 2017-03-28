#ifndef INSTRUMENT_NULL_HARDWARE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_NULL_HARDWARE_PLACE_MANAGEMENT_HPP


#include "InstrumentHardwarePlaceId.hpp"
#include "../api/InstrumentHardwarePlaceManagement.hpp"


namespace Instrument {
	inline hardware_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		return hardware_place_id_t(virtualCPUId);
	}
	
	inline hardware_place_id_t createdGPU()
	{
		return hardware_place_id_t();
	}
}


#endif // INSTRUMENT_NULL_HARDWARE_PLACE_MANAGEMENT_HPP
