#ifndef HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
#define HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP


#if HAVE_PAPI
#include "PAPI/PAPIHardwareCountersThreadLocalDataImplementation.hpp"
#else
#include "no-HC/NoHardwareCountersThreadLocalDataImplementation.hpp"
#endif


#endif // HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
