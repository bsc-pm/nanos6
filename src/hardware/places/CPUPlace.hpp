#ifndef CPU_PLACE_HPP
#define CPU_PLACE_HPP


#include "HardwarePlace.hpp"


class CPUPlace: public HardwarePlace {
public:
	CPUPlace (HardwarePlace *parent = nullptr)
		: HardwarePlace(parent)
	{
	}
	
};


#endif // CPU_PLACE_HPP
