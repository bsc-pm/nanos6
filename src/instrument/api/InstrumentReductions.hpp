/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_REDUCTIONS_HPP
#define INSTRUMENT_REDUCTIONS_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentDataAccessId.hpp>

class ReductionInfo;
class DataAccessRegion;


namespace Instrument {
	void allocatedReductionInfo(data_access_id_t dataAccessId, const ReductionInfo& reductionInfo, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
	void receivedCompatibleReductionInfo(data_access_id_t dataAccessId, const ReductionInfo& reductionInfo, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
	void deallocatedReductionInfo(data_access_id_t dataAccessId, const ReductionInfo *reductionInfo, const DataAccessRegion& originalRegion, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
	
	void initializedPrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
	void retrievedPrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
	void combinedPrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_REDUCTIONS_HPP
