/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef BOTTOM_MAP_ENTRY_HPP
#define BOTTOM_MAP_ENTRY_HPP

#include "DataAccess.hpp"
#include "ReductionInfo.hpp"

struct BottomMapEntry {
	DataAccess * _access;
	bool _satisfied;
	ReductionInfo * _reductionInfo;

	BottomMapEntry(DataAccess * access) :
	_access(access),
	_satisfied(true),
	_reductionInfo(nullptr)
	{
	}
};


#endif // BOTTOM_MAP_ENTRY_HPP
