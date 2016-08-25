#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP

#include "HardwarePlace.hpp"

class ComputePlace : public HardwarePlace {
private:
	friend class MemoryPlace;
	friend class Loader;
	friend class Machine;	

	ComputePlace(int index, HardwarePlace *parent = nullptr)
		: HardwarePlace(index, parent) 
	{
	}		
public:

};

#endif //COMPUTE_PLACE_HPP
