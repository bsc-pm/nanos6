/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "InstrumentExtrae.hpp"
#include "../api/InstrumentComputePlaceManagement.hpp"


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		if (!_traceAsThreads) {
			Extrae_change_num_threads(extrae_nanos_get_num_cpus_and_external_threads());
		}
		
		return compute_place_id_t(virtualCPUId);
	}
	
	inline compute_place_id_t createdGPU()
	{
		return compute_place_id_t();
	}
}


#endif // INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP
