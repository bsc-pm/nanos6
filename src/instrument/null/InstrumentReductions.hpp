/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_REDUCTIONS_HPP
#define INSTRUMENT_NULL_REDUCTIONS_HPP


#include "../api/InstrumentReductions.hpp"



namespace Instrument {
	inline void receivedCompatibleReductionInfo(
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void deallocatedReductionInfo(
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) const ReductionInfo *reductionInfo,
		__attribute__((unused)) const DataAccessRegion& originalRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	
	inline void enterAllocateReductionInfo(
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) const DataAccessRegion& accessRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void exitAllocateReductionInfo(
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void enterAllocatePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void exitAllocatePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void enterRetrievePrivateReductionStorage(
		__attribute__((unused)) const DataAccessRegion& originalRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void exitRetrievePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const DataAccessRegion& originalRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void enterInitializePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void exitInitializePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void enterCombinePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const DataAccessRegion& originalRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
	inline void exitCombinePrivateReductionStorage(
		__attribute__((unused)) const ReductionInfo& reductionInfo,
		__attribute__((unused)) const DataAccessRegion& privateStorage,
		__attribute__((unused)) const DataAccessRegion& originalRegion,
		__attribute__((unused)) const InstrumentationContext &context
	) {
	}
}


#endif // INSTRUMENT_NULL_REDUCTIONS_HPP
