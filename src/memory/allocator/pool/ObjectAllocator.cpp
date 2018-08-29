/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <ObjectAllocator.hpp>

template<> ObjectAllocator<DataAccess>::inner_type *ObjectAllocator<DataAccess>::_cache = nullptr;
template<> ObjectAllocator<ReductionInfo>::inner_type *ObjectAllocator<ReductionInfo>::_cache = nullptr;
template<> ObjectAllocator<BottomMapEntry>::inner_type *ObjectAllocator<BottomMapEntry>::_cache = nullptr;
