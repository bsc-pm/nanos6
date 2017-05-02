#ifndef HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
#define HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP


#if HAVE_PAPI
#include "PAPI/PAPIHardwareCountersThreadLocalData.hpp"
#else
#include "no-HC/NoHardwareCountersThreadLocalData.hpp"
#endif


#endif // HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
