/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef BOTTOM_MAP_ENTRY_HPP
#define BOTTOM_MAP_ENTRY_HPP

#include "DataAccess.hpp"
#include "ReductionInfo.hpp"

typedef struct BottomMapEntry {
	DataAccess * access;
	bool satisfied;
	ReductionInfo * reductionInfo;

	BottomMapEntry(DataAccess * accessN, __attribute__((unused)) Task * currentTask) : 
	access(accessN), 
	satisfied(true), 
	reductionInfo(nullptr) 
	{
	}
} BottomMapEntry;


#endif // BOTTOM_MAP_ENTRY_HPP
