
hardwareCountersDefinitions = [
    # PQoS events
    ("PQOS_MON_EVENT_L3_OCCUP"              , 4000000, "PQOS_MON_EVENT_L3_OCCUP              [LLC Usage]"),
    ("PQOS_MON_EVENT_LMEM_BW"               , 4000001, "PQOS_MON_EVENT_LMEM_BW               [Local Memory Bandwidth]"),
    ("PQOS_MON_EVENT_RMEM_BW"               , 4000002, "PQOS_MON_EVENT_RMEM_BW               [Remote Memory Bandwidth]"),
    ("PQOS_PERF_EVENT_LLC_MISS"             , 4000003, "PQOS_PERF_EVENT_LLC_MISS             [LLC Misses]"),
    ("PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS" , 4000004, "PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS [Retired Instructions]"),
    ("PQOS_PERF_EVENT_UNHALTED_CYCLES"      , 4000005, "PQOS_PERF_EVENT_UNHALTED_CYCLES      [Unhalted cycles]"),

    # PAPI events
    ("PAPI_L1_DCM"                          , 4100000, "PAPI_L1_DCM  [Level 1 data cache misses]"),
    ("PAPI_L1_ICM"                          , 4100001, "PAPI_L1_ICM  [Level 1 instruction cache misses]"),
    ("PAPI_L2_DCM"                          , 4100002, "PAPI_L2_DCM  [Level 2 data cache misses]"),
    ("PAPI_L2_ICM"                          , 4100003, "PAPI_L2_ICM  [Level 2 instruction cache misses]"),
    ("PAPI_L3_DCM"                          , 4100004, "PAPI_L3_DCM  [Level 3 data cache misses]"),
    ("PAPI_L3_ICM"                          , 4100005, "PAPI_L3_ICM  [Level 3 instruction cache misses]"),
    ("PAPI_L1_TCM"                          , 4100006, "PAPI_L1_TCM  [Level 1 cache misses]"),
    ("PAPI_L2_TCM"                          , 4100007, "PAPI_L2_TCM  [Level 2 cache misses]"),
    ("PAPI_L3_TCM"                          , 4100008, "PAPI_L3_TCM  [Level 3 cache misses]"),
    ("PAPI_CA_SNP"                          , 4100009, "PAPI_CA_SNP  [Requests for a snoop]"),
    ("PAPI_CA_SHR"                          , 4100010, "PAPI_CA_SHR  [Requests for exclusive access to shared cache line]"),
    ("PAPI_CA_CLN"                          , 4100011, "PAPI_CA_CLN  [Requests for exclusive access to clean cache line]"),
    ("PAPI_CA_INV"                          , 4100012, "PAPI_CA_INV  [Requests for cache line invalidation]"),
    ("PAPI_CA_ITV"                          , 4100013, "PAPI_CA_ITV  [Requests for cache line intervention]"),
    ("PAPI_L3_LDM"                          , 4100014, "PAPI_L3_LDM  [Level 3 load misses]"),
    ("PAPI_L3_STM"                          , 4100015, "PAPI_L3_STM  [Level 3 store misses]"),
    ("PAPI_BRU_IDL"                         , 4100016, "PAPI_BRU_IDL [Cycles branch units are idle]"),
    ("PAPI_FXU_IDL"                         , 4100017, "PAPI_FXU_IDL [Cycles integer units are idle]"),
    ("PAPI_FPU_IDL"                         , 4100018, "PAPI_FPU_IDL [Cycles floating point units are idle]"),
    ("PAPI_LSU_IDL"                         , 4100019, "PAPI_LSU_IDL [Cycles load/store units are idle]"),
    ("PAPI_TLB_DM"                          , 4100020, "PAPI_TLB_DM  [Data translation lookaside buffer misses]"),
    ("PAPI_TLB_IM"                          , 4100021, "PAPI_TLB_IM  [Instruction translation lookaside buffer misses]"),
    ("PAPI_TLB_TL"                          , 4100022, "PAPI_TLB_TL  [Total translation lookaside buffer misses]"),
    ("PAPI_L1_LDM"                          , 4100023, "PAPI_L1_LDM  [Level 1 load misses]"),
    ("PAPI_L1_STM"                          , 4100024, "PAPI_L1_STM  [Level 1 store misses]"),
    ("PAPI_L2_LDM"                          , 4100025, "PAPI_L2_LDM  [Level 2 load misses]"),
    ("PAPI_L2_STM"                          , 4100026, "PAPI_L2_STM  [Level 2 store misses]"),
    ("PAPI_BTAC_M"                          , 4100027, "PAPI_BTAC_M  [Branch target address cache misses]"),
    ("PAPI_PRF_DM"                          , 4100028, "PAPI_PRF_DM  [Data prefetch cache misses]"),
    ("PAPI_L3_DCH"                          , 4100029, "PAPI_L3_DCH  [Level 3 data cache hits]"),
    ("PAPI_TLB_SD"                          , 4100030, "PAPI_TLB_SD  [Translation lookaside buffer shootdowns]"),
    ("PAPI_CSR_FAL"                         , 4100031, "PAPI_CSR_FAL [Failed store conditional instructions]"),
    ("PAPI_CSR_SUC"                         , 4100032, "PAPI_CSR_SUC [Successful store conditional instructions]"),
    ("PAPI_CSR_TOT"                         , 4100033, "PAPI_CSR_TOT [Total store conditional instructions]"),
    ("PAPI_MEM_SCY"                         , 4100034, "PAPI_MEM_SCY [Cycles Stalled Waiting for memory accesses]"),
    ("PAPI_MEM_RCY"                         , 4100035, "PAPI_MEM_RCY [Cycles Stalled Waiting for memory Reads]"),
    ("PAPI_MEM_WCY"                         , 4100036, "PAPI_MEM_WCY [Cycles Stalled Waiting for memory writes]"),
    ("PAPI_STL_ICY"                         , 4100037, "PAPI_STL_ICY [Cycles with no instruction issue]"),
    ("PAPI_FUL_ICY"                         , 4100038, "PAPI_FUL_ICY [Cycles with maximum instruction issue]"),
    ("PAPI_STL_CCY"                         , 4100039, "PAPI_STL_CCY [Cycles with no instructions completed]"),
    ("PAPI_FUL_CCY"                         , 4100040, "PAPI_FUL_CCY [Cycles with maximum instructions completed]"),
    ("PAPI_HW_INT"                          , 4100041, "PAPI_HW_INT  [Hardware interrupts]"),
    ("PAPI_BR_UCN"                          , 4100042, "PAPI_BR_UCN  [Unconditional branch instructions]"),
    ("PAPI_BR_CN"                           , 4100043, "PAPI_BR_CN   [Conditional branch instructions]"),
    ("PAPI_BR_TKN"                          , 4100044, "PAPI_BR_TKN  [Conditional branch instructions taken]"),
    ("PAPI_BR_NTK"                          , 4100045, "PAPI_BR_NTK  [Conditional branch instructions not taken]"),
    ("PAPI_BR_MSP"                          , 4100046, "PAPI_BR_MSP  [Conditional branch instructions mispredicted]"),
    ("PAPI_BR_PRC"                          , 4100047, "PAPI_BR_PRC  [Conditional branch instructions correctly predicted]"),
    ("PAPI_FMA_INS"                         , 4100048, "PAPI_FMA_INS [FMA instructions completed]"),
    ("PAPI_TOT_IIS"                         , 4100049, "PAPI_TOT_IIS [Instructions issued]"),
    ("PAPI_TOT_INS"                         , 4100050, "PAPI_TOT_INS [Instructions completed]"),
    ("PAPI_INT_INS"                         , 4100051, "PAPI_INT_INS [Integer instructions]"),
    ("PAPI_FP_INS"                          , 4100052, "PAPI_FP_INS  [Floating point instructions]"),
    ("PAPI_LD_INS"                          , 4100053, "PAPI_LD_INS  [Load instructions]"),
    ("PAPI_SR_INS"                          , 4100054, "PAPI_SR_INS  [Store instructions]"),
    ("PAPI_BR_INS"                          , 4100055, "PAPI_BR_INS  [Branch instructions]"),
    ("PAPI_VEC_INS"                         , 4100056, "PAPI_VEC_INS [Vector/SIMD instructions (could include integer)]"),
    ("PAPI_RES_STL"                         , 4100057, "PAPI_RES_STL [Cycles stalled on any resource]"),
    ("PAPI_FP_STAL"                         , 4100058, "PAPI_FP_STAL [Cycles the FP unit(s) are stalled]"),
    ("PAPI_TOT_CYC"                         , 4100059, "PAPI_TOT_CYC [Total cycles]"),
    ("PAPI_LST_INS"                         , 4100060, "PAPI_LST_INS [Load/store instructions completed]"),
    ("PAPI_SYC_INS"                         , 4100061, "PAPI_SYC_INS [Synchronization instructions completed]"),
    ("PAPI_L1_DCH"                          , 4100062, "PAPI_L1_DCH  [Level 1 data cache hits]"),
    ("PAPI_L2_DCH"                          , 4100063, "PAPI_L2_DCH  [Level 2 data cache hits]"),
    ("PAPI_L1_DCA"                          , 4100064, "PAPI_L1_DCA  [Level 1 data cache accesses]"),
    ("PAPI_L2_DCA"                          , 4100065, "PAPI_L2_DCA  [Level 2 data cache accesses]"),
    ("PAPI_L3_DCA"                          , 4100066, "PAPI_L3_DCA  [Level 3 data cache accesses]"),
    ("PAPI_L1_DCR"                          , 4100067, "PAPI_L1_DCR  [Level 1 data cache reads]"),
    ("PAPI_L2_DCR"                          , 4100068, "PAPI_L2_DCR  [Level 2 data cache reads]"),
    ("PAPI_L3_DCR"                          , 4100069, "PAPI_L3_DCR  [Level 3 data cache reads]"),
    ("PAPI_L1_DCW"                          , 4100070, "PAPI_L1_DCW  [Level 1 data cache writes]"),
    ("PAPI_L2_DCW"                          , 4100071, "PAPI_L2_DCW  [Level 2 data cache writes]"),
    ("PAPI_L3_DCW"                          , 4100072, "PAPI_L3_DCW  [Level 3 data cache writes]"),
    ("PAPI_L1_ICH"                          , 4100073, "PAPI_L1_ICH  [Level 1 instruction cache hits]"),
    ("PAPI_L2_ICH"                          , 4100074, "PAPI_L2_ICH  [Level 2 instruction cache hits]"),
    ("PAPI_L3_ICH"                          , 4100075, "PAPI_L3_ICH  [Level 3 instruction cache hits]"),
    ("PAPI_L1_ICA"                          , 4100076, "PAPI_L1_ICA  [Level 1 instruction cache accesses]"),
    ("PAPI_L2_ICA"                          , 4100077, "PAPI_L2_ICA  [Level 2 instruction cache accesses]"),
    ("PAPI_L3_ICA"                          , 4100078, "PAPI_L3_ICA  [Level 3 instruction cache accesses]"),
    ("PAPI_L1_ICR"                          , 4100079, "PAPI_L1_ICR  [Level 1 instruction cache reads]"),
    ("PAPI_L2_ICR"                          , 4100080, "PAPI_L2_ICR  [Level 2 instruction cache reads]"),
    ("PAPI_L3_ICR"                          , 4100081, "PAPI_L3_ICR  [Level 3 instruction cache reads]"),
    ("PAPI_L1_ICW"                          , 4100082, "PAPI_L1_ICW  [Level 1 instruction cache writes]"),
    ("PAPI_L2_ICW"                          , 4100083, "PAPI_L2_ICW  [Level 2 instruction cache writes]"),
    ("PAPI_L3_ICW"                          , 4100084, "PAPI_L3_ICW  [Level 3 instruction cache writes]"),
    ("PAPI_L1_TCH"                          , 4100085, "PAPI_L1_TCH  [Level 1 total cache hits]"),
    ("PAPI_L2_TCH"                          , 4100086, "PAPI_L2_TCH  [Level 2 total cache hits]"),
    ("PAPI_L3_TCH"                          , 4100087, "PAPI_L3_TCH  [Level 3 total cache hits]"),
    ("PAPI_L1_TCA"                          , 4100088, "PAPI_L1_TCA  [Level 1 total cache accesses]"),
    ("PAPI_L2_TCA"                          , 4100089, "PAPI_L2_TCA  [Level 2 total cache accesses]"),
    ("PAPI_L3_TCA"                          , 4100090, "PAPI_L3_TCA  [Level 3 total cache accesses]"),
    ("PAPI_L1_TCR"                          , 4100091, "PAPI_L1_TCR  [Level 1 total cache reads]"),
    ("PAPI_L2_TCR"                          , 4100092, "PAPI_L2_TCR  [Level 2 total cache reads]"),
    ("PAPI_L3_TCR"                          , 4100093, "PAPI_L3_TCR  [Level 3 total cache reads]"),
    ("PAPI_L1_TCW"                          , 4100094, "PAPI_L1_TCW  [Level 1 total cache writes]"),
    ("PAPI_L2_TCW"                          , 4100095, "PAPI_L2_TCW  [Level 2 total cache writes]"),
    ("PAPI_L3_TCW"                          , 4100096, "PAPI_L3_TCW  [Level 3 total cache writes]"),
    ("PAPI_FML_INS"                         , 4100097, "PAPI_FML_INS [Floating point multiply instructions]"),
    ("PAPI_FAD_INS"                         , 4100098, "PAPI_FAD_INS [Floating point add instructions]"),
    ("PAPI_FDV_INS"                         , 4100099, "PAPI_FDV_INS [Floating point divide instructions]"),
    ("PAPI_FSQ_INS"                         , 4100100, "PAPI_FSQ_INS [Floating point square root instructions]"),
    ("PAPI_FNV_INS"                         , 4100101, "PAPI_FNV_INS [Floating point inverse instructions]"),
    ("PAPI_FP_OPS"                          , 4100102, "PAPI_FP_OPS  [Floating point operations]"),
    ("PAPI_SP_OPS"                          , 4100103, "PAPI_SP_OPS  [Floating point operations; optimized to count scaled single precision vector operations]"),
    ("PAPI_DP_OPS"                          , 4100104, "PAPI_DP_OPS  [Floating point operations; optimized to count scaled double precision vector operations]"),
    ("PAPI_VEC_SP"                          , 4100105, "PAPI_VEC_SP  [Single precision vector/SIMD instructions]"),
    ("PAPI_VEC_DP"                          , 4100106, "PAPI_VEC_DP  [Double precision vector/SIMD instructions]"),
    ("PAPI_REF_CYC"                         , 4100107, "PAPI_REF_CYC [Reference clock cycles]"),
]
