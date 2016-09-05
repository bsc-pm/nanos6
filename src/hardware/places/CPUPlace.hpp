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
