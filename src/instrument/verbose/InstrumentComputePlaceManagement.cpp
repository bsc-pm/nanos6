/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/


#include "InstrumentComputePlaceManagement.hpp"
#include "InstrumentVerbose.hpp"


namespace Instrument {
	compute_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		return compute_place_id_t(virtualCPUId, Verbose::_concurrentUnorderedListSlotManager.getSlot());
	}
	
	compute_place_id_t createdCUDAGPU()
	{
		return compute_place_id_t(-2, Verbose::_concurrentUnorderedListSlotManager.getSlot());
	}
}

