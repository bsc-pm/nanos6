/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "RuntimeInfo.hpp"



SpinLock RuntimeInfo::_lock;
std::vector<nanos6_runtime_info_entry_t> RuntimeInfo::_contents;

