#include "RuntimeInfo.hpp"



SpinLock RuntimeInfo::_lock;
std::vector<nanos6_runtime_info_entry_t> RuntimeInfo::_contents;

