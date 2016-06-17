#ifndef CPU_PLACE_HPP
#define CPU_PLACE_HPP

#include "HardwarePlace.hpp"

class CPUPlace: public HardwarePlace {
public:
	CPUPlace (int index, HardwarePlace *parent = nullptr)
		: HardwarePlace(index, parent)
	{
	}
	
};

#endif // CPU_PLACE_HPP
