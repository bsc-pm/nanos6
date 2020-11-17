/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUPPORTED_HARDWARE_COUNTERS_HPP
#define SUPPORTED_HARDWARE_COUNTERS_HPP

#include <cstdint>
#include <map>

namespace HWCounters {

	enum backends_t {
		PAPI_BACKEND = 0,
		PQOS_BACKEND,
		RAPL_BACKEND,
		NUM_BACKENDS
	};

	// NOTE: To add new events, do as follows:
	// - If the event added is from an existing backend (e.g. PQOS):
	// -- 1) Add the new event before the current maximum event (PQOS_MAX_EVENT)
	// -- 2) The identifier of the new event should be the previous maximum event + 1
	// -- 3) Update PQOS_MAX_EVENT
	// -- WARNING: Be aware not to create collisions between backend identifiers!
	//    (i.e., PAPI takes more than 100 event ids, so the next backend should start
	//    with 400 instead of 300)
	//
	// - If the event added is from a new backend:
	// -- 1) Add the new backend to the previous enum (backends_t)
	// -- 2) Following the observed pattern, add "MIN", "MAX", and "NUM" variables for
	//       the new backend, as well as any needed event identifier
	// -- 3) The identifiers should start with the previous minimum event backend + 100
	//       (i.e., NEWBACKEND_MIN_EVENT = PAPI_MIN_EVENT (200) + 100
	// -- 4) Add the MAX variable to the total number of events
	//       (i.e., TOTAL_NUM_EVENTS = ... + NEWBACKEND_NUM_EVENTS
	//
	// In all cases: Add a description of the event below (counterDescriptions)
	enum counters_t {
		//    PQOS EVENTS    //
		HWC_PQOS_MIN_EVENT = 100,                         // PQOS: Minimum event id
		HWC_PQOS_MON_EVENT_L3_OCCUP = HWC_PQOS_MIN_EVENT, // PQOS: LLC Usage
		HWC_PQOS_MON_EVENT_LMEM_BW,                       // PQOS: Local Memory Bandwidth
		HWC_PQOS_MON_EVENT_RMEM_BW,                       // PQOS: Remote Memory Bandwidth
		HWC_PQOS_PERF_EVENT_LLC_MISS,                     // PQOS: LLC Misses
		HWC_PQOS_PERF_EVENT_INSTRUCTIONS,                 // PQOS: Retired Instructions
		HWC_PQOS_PERF_EVENT_CYCLES,                       // PQOS: Unhalted cycles
		HWC_PQOS_MAX_EVENT = HWC_PQOS_PERF_EVENT_CYCLES,  // PQOS: Maximum event id
		HWC_PQOS_NUM_EVENTS = HWC_PQOS_MAX_EVENT - HWC_PQOS_MIN_EVENT + 1,
		//    PAPI EVENTS    //
		HWC_PAPI_MIN_EVENT = 200,             // PAPI: Minimum event id
		HWC_PAPI_L1_DCM = HWC_PAPI_MIN_EVENT, // PAPI: Level 1 data cache misses
		HWC_PAPI_L1_ICM,                      // PAPI: Level 1 instruction cache misses
		HWC_PAPI_L2_DCM,                      // PAPI: Level 2 data cache misses
		HWC_PAPI_L2_ICM,                      // PAPI: Level 2 instruction cache misses
		HWC_PAPI_L3_DCM,                      // PAPI: Level 3 data cache misses
		HWC_PAPI_L3_ICM,                      // PAPI: Level 3 instruction cache misses
		HWC_PAPI_L1_TCM,                      // PAPI: Level 1 cache misses
		HWC_PAPI_L2_TCM,                      // PAPI: Level 2 cache misses
		HWC_PAPI_L3_TCM,                      // PAPI: Level 3 cache misses
		HWC_PAPI_CA_SNP,                      // PAPI: Requests for a snoop
		HWC_PAPI_CA_SHR,                      // PAPI: Requests for exclusive access to shared cache line
		HWC_PAPI_CA_CLN,                      // PAPI: Requests for exclusive access to clean cache line
		HWC_PAPI_CA_INV,                      // PAPI: Requests for cache line invalidation
		HWC_PAPI_CA_ITV,                      // PAPI: Requests for cache line intervention
		HWC_PAPI_L3_LDM,                      // PAPI: Level 3 load misses
		HWC_PAPI_L3_STM,                      // PAPI: Level 3 store misses
		HWC_PAPI_BRU_IDL,                     // PAPI: Cycles branch units are idle
		HWC_PAPI_FXU_IDL,                     // PAPI: Cycles integer units are idle
		HWC_PAPI_FPU_IDL,                     // PAPI: Cycles floating point units are idle
		HWC_PAPI_LSU_IDL,                     // PAPI: Cycles load/store units are idle
		HWC_PAPI_TLB_DM,                      // PAPI: Data translation lookaside buffer misses
		HWC_PAPI_TLB_IM,                      // PAPI: Instruction translation lookaside buffer misses
		HWC_PAPI_TLB_TL,                      // PAPI: Total translation lookaside buffer misses
		HWC_PAPI_L1_LDM,                      // PAPI: Level 1 load misses
		HWC_PAPI_L1_STM,                      // PAPI: Level 1 store misses
		HWC_PAPI_L2_LDM,                      // PAPI: Level 2 load misses
		HWC_PAPI_L2_STM,                      // PAPI: Level 2 store misses
		HWC_PAPI_BTAC_M,                      // PAPI: Branch target address cache misses
		HWC_PAPI_PRF_DM,                      // PAPI: Data prefetch cache misses
		HWC_PAPI_L3_DCH,                      // PAPI: Level 3 data cache hits
		HWC_PAPI_TLB_SD,                      // PAPI: Translation lookaside buffer shootdowns
		HWC_PAPI_CSR_FAL,                     // PAPI: Failed store conditional instructions
		HWC_PAPI_CSR_SUC,                     // PAPI: Successful store conditional instructions
		HWC_PAPI_CSR_TOT,                     // PAPI: Total store conditional instructions
		HWC_PAPI_MEM_SCY,                     // PAPI: Cycles Stalled Waiting for memory accesses
		HWC_PAPI_MEM_RCY,                     // PAPI: Cycles Stalled Waiting for memory Reads
		HWC_PAPI_MEM_WCY,                     // PAPI: Cycles Stalled Waiting for memory writes
		HWC_PAPI_STL_ICY,                     // PAPI: Cycles with no instruction issue
		HWC_PAPI_FUL_ICY,                     // PAPI: Cycles with maximum instruction issue
		HWC_PAPI_STL_CCY,                     // PAPI: Cycles with no instructions completed
		HWC_PAPI_FUL_CCY,                     // PAPI: Cycles with maximum instructions completed
		HWC_PAPI_HW_INT,                      // PAPI: Hardware interrupts
		HWC_PAPI_BR_UCN,                      // PAPI: Unconditional branch instructions
		HWC_PAPI_BR_CN,                       // PAPI: Conditional branch instructions
		HWC_PAPI_BR_TKN,                      // PAPI: Conditional branch instructions taken
		HWC_PAPI_BR_NTK,                      // PAPI: Conditional branch instructions not taken
		HWC_PAPI_BR_MSP,                      // PAPI: Conditional branch instructions mispredicted
		HWC_PAPI_BR_PRC,                      // PAPI: Conditional branch instructions correctly predicted
		HWC_PAPI_FMA_INS,                     // PAPI: FMA instructions completed
		HWC_PAPI_TOT_IIS,                     // PAPI: Instructions issued
		HWC_PAPI_TOT_INS,                     // PAPI: Instructions completed
		HWC_PAPI_INT_INS,                     // PAPI: Integer instructions
		HWC_PAPI_FP_INS,                      // PAPI: Floating point instructions
		HWC_PAPI_LD_INS,                      // PAPI: Load instructions
		HWC_PAPI_SR_INS,                      // PAPI: Store instructions
		HWC_PAPI_BR_INS,                      // PAPI: Branch instructions
		HWC_PAPI_VEC_INS,                     // PAPI: Vector/SIMD instructions (could include integer)
		HWC_PAPI_RES_STL,                     // PAPI: Cycles stalled on any resource
		HWC_PAPI_FP_STAL,                     // PAPI: Cycles the FP unit(s) are stalled
		HWC_PAPI_TOT_CYC,                     // PAPI: Total cycles
		HWC_PAPI_LST_INS,                     // PAPI: Load/store instructions completed
		HWC_PAPI_SYC_INS,                     // PAPI: Synchronization instructions completed
		HWC_PAPI_L1_DCH,                      // PAPI: Level 1 data cache hits
		HWC_PAPI_L2_DCH,                      // PAPI: Level 2 data cache hits
		HWC_PAPI_L1_DCA,                      // PAPI: Level 1 data cache accesses
		HWC_PAPI_L2_DCA,                      // PAPI: Level 2 data cache accesses
		HWC_PAPI_L3_DCA,                      // PAPI: Level 3 data cache accesses
		HWC_PAPI_L1_DCR,                      // PAPI: Level 1 data cache reads
		HWC_PAPI_L2_DCR,                      // PAPI: Level 2 data cache reads
		HWC_PAPI_L3_DCR,                      // PAPI: Level 3 data cache reads
		HWC_PAPI_L1_DCW,                      // PAPI: Level 1 data cache writes
		HWC_PAPI_L2_DCW,                      // PAPI: Level 2 data cache writes
		HWC_PAPI_L3_DCW,                      // PAPI: Level 3 data cache writes
		HWC_PAPI_L1_ICH,                      // PAPI: Level 1 instruction cache hits
		HWC_PAPI_L2_ICH,                      // PAPI: Level 2 instruction cache hits
		HWC_PAPI_L3_ICH,                      // PAPI: Level 3 instruction cache hits
		HWC_PAPI_L1_ICA,                      // PAPI: Level 1 instruction cache accesses
		HWC_PAPI_L2_ICA,                      // PAPI: Level 2 instruction cache accesses
		HWC_PAPI_L3_ICA,                      // PAPI: Level 3 instruction cache accesses
		HWC_PAPI_L1_ICR,                      // PAPI: Level 1 instruction cache reads
		HWC_PAPI_L2_ICR,                      // PAPI: Level 2 instruction cache reads
		HWC_PAPI_L3_ICR,                      // PAPI: Level 3 instruction cache reads
		HWC_PAPI_L1_ICW,                      // PAPI: Level 1 instruction cache writes
		HWC_PAPI_L2_ICW,                      // PAPI: Level 2 instruction cache writes
		HWC_PAPI_L3_ICW,                      // PAPI: Level 3 instruction cache writes
		HWC_PAPI_L1_TCH,                      // PAPI: Level 1 total cache hits
		HWC_PAPI_L2_TCH,                      // PAPI: Level 2 total cache hits
		HWC_PAPI_L3_TCH,                      // PAPI: Level 3 total cache hits
		HWC_PAPI_L1_TCA,                      // PAPI: Level 1 total cache accesses
		HWC_PAPI_L2_TCA,                      // PAPI: Level 2 total cache accesses
		HWC_PAPI_L3_TCA,                      // PAPI: Level 3 total cache accesses
		HWC_PAPI_L1_TCR,                      // PAPI: Level 1 total cache reads
		HWC_PAPI_L2_TCR,                      // PAPI: Level 2 total cache reads
		HWC_PAPI_L3_TCR,                      // PAPI: Level 3 total cache reads
		HWC_PAPI_L1_TCW,                      // PAPI: Level 1 total cache writes
		HWC_PAPI_L2_TCW,                      // PAPI: Level 2 total cache writes
		HWC_PAPI_L3_TCW,                      // PAPI: Level 3 total cache writes
		HWC_PAPI_FML_INS,                     // PAPI: Floating point multiply instructions
		HWC_PAPI_FAD_INS,                     // PAPI: Floating point add instructions
		HWC_PAPI_FDV_INS,                     // PAPI: Floating point divide instructions
		HWC_PAPI_FSQ_INS,                     // PAPI: Floating point square root instructions
		HWC_PAPI_FNV_INS,                     // PAPI: Floating point inverse instructions
		HWC_PAPI_FP_OPS,                      // PAPI: Floating point operations
		HWC_PAPI_SP_OPS,                      // PAPI: Floating point operations; optimized to count scaled single precision vector operations
		HWC_PAPI_DP_OPS,                      // PAPI: Floating point operations; optimized to count scaled double precision vector operations
		HWC_PAPI_VEC_SP,                      // PAPI: Single precision vector/SIMD instructions
		HWC_PAPI_VEC_DP,                      // PAPI: Double precision vector/SIMD instructions
		HWC_PAPI_REF_CYC,                     // PAPI: Reference clock cycles
		HWC_PAPI_MAX_EVENT = HWC_PAPI_REF_CYC,
		HWC_PAPI_NUM_EVENTS = HWC_PAPI_MAX_EVENT - HWC_PAPI_MIN_EVENT + 1,
		//    GENERAL    //
		HWC_TOTAL_NUM_EVENTS = HWC_PQOS_NUM_EVENTS + HWC_PAPI_NUM_EVENTS
	};

	static std::map<uint64_t, const char* const> counterDescriptions = {
		{HWC_PQOS_MON_EVENT_L3_OCCUP      , "PQOS_MON_EVENT_L3_OCCUP"},
		{HWC_PQOS_MON_EVENT_LMEM_BW       , "PQOS_MON_EVENT_LMEM_BW"},
		{HWC_PQOS_MON_EVENT_RMEM_BW       , "PQOS_MON_EVENT_RMEM_BW"},
		{HWC_PQOS_PERF_EVENT_LLC_MISS     , "PQOS_PERF_EVENT_LLC_MISS"},
		{HWC_PQOS_PERF_EVENT_INSTRUCTIONS , "PQOS_PERF_EVENT_INSTRUCTIONS"},
		{HWC_PQOS_PERF_EVENT_CYCLES       , "PQOS_PERF_EVENT_CYCLES"},
		{HWC_PAPI_L1_DCM                  , "PAPI_L1_DCM"},
		{HWC_PAPI_L1_ICM                  , "PAPI_L1_ICM"},
		{HWC_PAPI_L2_DCM                  , "PAPI_L2_DCM"},
		{HWC_PAPI_L2_ICM                  , "PAPI_L2_ICM"},
		{HWC_PAPI_L3_DCM                  , "PAPI_L3_DCM"},
		{HWC_PAPI_L3_ICM                  , "PAPI_L3_ICM"},
		{HWC_PAPI_L1_TCM                  , "PAPI_L1_TCM"},
		{HWC_PAPI_L2_TCM                  , "PAPI_L2_TCM"},
		{HWC_PAPI_L3_TCM                  , "PAPI_L3_TCM"},
		{HWC_PAPI_CA_SNP                  , "PAPI_CA_SNP"},
		{HWC_PAPI_CA_SHR                  , "PAPI_CA_SHR"},
		{HWC_PAPI_CA_CLN                  , "PAPI_CA_CLN"},
		{HWC_PAPI_CA_INV                  , "PAPI_CA_INV"},
		{HWC_PAPI_CA_ITV                  , "PAPI_CA_ITV"},
		{HWC_PAPI_L3_LDM                  , "PAPI_L3_LDM"},
		{HWC_PAPI_L3_STM                  , "PAPI_L3_STM"},
		{HWC_PAPI_BRU_IDL                 , "PAPI_BRU_IDL"},
		{HWC_PAPI_FXU_IDL                 , "PAPI_FXU_IDL"},
		{HWC_PAPI_FPU_IDL                 , "PAPI_FPU_IDL"},
		{HWC_PAPI_LSU_IDL                 , "PAPI_LSU_IDL"},
		{HWC_PAPI_TLB_DM                  , "PAPI_TLB_DM"},
		{HWC_PAPI_TLB_IM                  , "PAPI_TLB_IM"},
		{HWC_PAPI_TLB_TL                  , "PAPI_TLB_TL"},
		{HWC_PAPI_L1_LDM                  , "PAPI_L1_LDM"},
		{HWC_PAPI_L1_STM                  , "PAPI_L1_STM"},
		{HWC_PAPI_L2_LDM                  , "PAPI_L2_LDM"},
		{HWC_PAPI_L2_STM                  , "PAPI_L2_STM"},
		{HWC_PAPI_BTAC_M                  , "PAPI_BTAC_M"},
		{HWC_PAPI_PRF_DM                  , "PAPI_PRF_DM"},
		{HWC_PAPI_L3_DCH                  , "PAPI_L3_DCH"},
		{HWC_PAPI_TLB_SD                  , "PAPI_TLB_SD"},
		{HWC_PAPI_CSR_FAL                 , "PAPI_CSR_FAL"},
		{HWC_PAPI_CSR_SUC                 , "PAPI_CSR_SUC"},
		{HWC_PAPI_CSR_TOT                 , "PAPI_CSR_TOT"},
		{HWC_PAPI_MEM_SCY                 , "PAPI_MEM_SCY"},
		{HWC_PAPI_MEM_RCY                 , "PAPI_MEM_RCY"},
		{HWC_PAPI_MEM_WCY                 , "PAPI_MEM_WCY"},
		{HWC_PAPI_STL_ICY                 , "PAPI_STL_ICY"},
		{HWC_PAPI_FUL_ICY                 , "PAPI_FUL_ICY"},
		{HWC_PAPI_STL_CCY                 , "PAPI_STL_CCY"},
		{HWC_PAPI_FUL_CCY                 , "PAPI_FUL_CCY"},
		{HWC_PAPI_HW_INT                  , "PAPI_HW_INT"},
		{HWC_PAPI_BR_UCN                  , "PAPI_BR_UCN"},
		{HWC_PAPI_BR_CN                   , "PAPI_BR_CN"},
		{HWC_PAPI_BR_TKN                  , "PAPI_BR_TKN"},
		{HWC_PAPI_BR_NTK                  , "PAPI_BR_NTK"},
		{HWC_PAPI_BR_MSP                  , "PAPI_BR_MSP"},
		{HWC_PAPI_BR_PRC                  , "PAPI_BR_PRC"},
		{HWC_PAPI_FMA_INS                 , "PAPI_FMA_INS"},
		{HWC_PAPI_TOT_IIS                 , "PAPI_TOT_IIS"},
		{HWC_PAPI_TOT_INS                 , "PAPI_TOT_INS"},
		{HWC_PAPI_INT_INS                 , "PAPI_INT_INS"},
		{HWC_PAPI_FP_INS                  , "PAPI_FP_INS"},
		{HWC_PAPI_LD_INS                  , "PAPI_LD_INS"},
		{HWC_PAPI_SR_INS                  , "PAPI_SR_INS"},
		{HWC_PAPI_BR_INS                  , "PAPI_BR_INS"},
		{HWC_PAPI_VEC_INS                 , "PAPI_VEC_INS"},
		{HWC_PAPI_RES_STL                 , "PAPI_RES_STL"},
		{HWC_PAPI_FP_STAL                 , "PAPI_FP_STAL"},
		{HWC_PAPI_TOT_CYC                 , "PAPI_TOT_CYC"},
		{HWC_PAPI_LST_INS                 , "PAPI_LST_INS"},
		{HWC_PAPI_SYC_INS                 , "PAPI_SYC_INS"},
		{HWC_PAPI_L1_DCH                  , "PAPI_L1_DCH"},
		{HWC_PAPI_L2_DCH                  , "PAPI_L2_DCH"},
		{HWC_PAPI_L1_DCA                  , "PAPI_L1_DCA"},
		{HWC_PAPI_L2_DCA                  , "PAPI_L2_DCA"},
		{HWC_PAPI_L3_DCA                  , "PAPI_L3_DCA"},
		{HWC_PAPI_L1_DCR                  , "PAPI_L1_DCR"},
		{HWC_PAPI_L2_DCR                  , "PAPI_L2_DCR"},
		{HWC_PAPI_L3_DCR                  , "PAPI_L3_DCR"},
		{HWC_PAPI_L1_DCW                  , "PAPI_L1_DCW"},
		{HWC_PAPI_L2_DCW                  , "PAPI_L2_DCW"},
		{HWC_PAPI_L3_DCW                  , "PAPI_L3_DCW"},
		{HWC_PAPI_L1_ICH                  , "PAPI_L1_ICH"},
		{HWC_PAPI_L2_ICH                  , "PAPI_L2_ICH"},
		{HWC_PAPI_L3_ICH                  , "PAPI_L3_ICH"},
		{HWC_PAPI_L1_ICA                  , "PAPI_L1_ICA"},
		{HWC_PAPI_L2_ICA                  , "PAPI_L2_ICA"},
		{HWC_PAPI_L3_ICA                  , "PAPI_L3_ICA"},
		{HWC_PAPI_L1_ICR                  , "PAPI_L1_ICR"},
		{HWC_PAPI_L2_ICR                  , "PAPI_L2_ICR"},
		{HWC_PAPI_L3_ICR                  , "PAPI_L3_ICR"},
		{HWC_PAPI_L1_ICW                  , "PAPI_L1_ICW"},
		{HWC_PAPI_L2_ICW                  , "PAPI_L2_ICW"},
		{HWC_PAPI_L3_ICW                  , "PAPI_L3_ICW"},
		{HWC_PAPI_L1_TCH                  , "PAPI_L1_TCH"},
		{HWC_PAPI_L2_TCH                  , "PAPI_L2_TCH"},
		{HWC_PAPI_L3_TCH                  , "PAPI_L3_TCH"},
		{HWC_PAPI_L1_TCA                  , "PAPI_L1_TCA"},
		{HWC_PAPI_L2_TCA                  , "PAPI_L2_TCA"},
		{HWC_PAPI_L3_TCA                  , "PAPI_L3_TCA"},
		{HWC_PAPI_L1_TCR                  , "PAPI_L1_TCR"},
		{HWC_PAPI_L2_TCR                  , "PAPI_L2_TCR"},
		{HWC_PAPI_L3_TCR                  , "PAPI_L3_TCR"},
		{HWC_PAPI_L1_TCW                  , "PAPI_L1_TCW"},
		{HWC_PAPI_L2_TCW                  , "PAPI_L2_TCW"},
		{HWC_PAPI_L3_TCW                  , "PAPI_L3_TCW"},
		{HWC_PAPI_FML_INS                 , "PAPI_FML_INS"},
		{HWC_PAPI_FAD_INS                 , "PAPI_FAD_INS"},
		{HWC_PAPI_FDV_INS                 , "PAPI_FDV_INS"},
		{HWC_PAPI_FSQ_INS                 , "PAPI_FSQ_INS"},
		{HWC_PAPI_FNV_INS                 , "PAPI_FNV_INS"},
		{HWC_PAPI_FP_OPS                  , "PAPI_FP_OPS"},
		{HWC_PAPI_SP_OPS                  , "PAPI_SP_OPS"},
		{HWC_PAPI_DP_OPS                  , "PAPI_DP_OPS"},
		{HWC_PAPI_VEC_SP                  , "PAPI_VEC_SP"},
		{HWC_PAPI_VEC_DP                  , "PAPI_VEC_DP"},
		{HWC_PAPI_REF_CYC                 , "PAPI_REF_CYC"}
	};
}

#endif // SUPPORTED_HARDWARE_COUNTERS_HPP
