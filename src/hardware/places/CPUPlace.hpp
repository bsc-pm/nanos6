/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_PLACE_HPP
#define CPU_PLACE_HPP

#include "ComputePlace.hpp"

class CPUPlace: public ComputePlace {
public:
	CPUPlace (int index = 0/*, ComputePlace *parent = nullptr*/)
		: ComputePlace(index/*, parent*/)
	{
	}
	
};

#endif // CPU_PLACE_HPP
