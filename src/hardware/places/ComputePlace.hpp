#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP

#include "HardwarePlace.hpp"

class ComputePlace : public HardwarePlace {
private:
	ComputePlace(int index, HardwarePlace *parent = nullptr)
		: HardwarePlace(index, parent) 
	{
	}		
public:

};

#endif //COMPUTE_PLACE_HPP
