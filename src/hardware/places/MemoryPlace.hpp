#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

#include <vector>
#include <map>
#include "../../memory/AddressSpace.hpp"

class ComputePlace;

class MemoryPlace {
private:
	typedef std::map<int, ComputePlace*> processingUnits_t;
	processingUnits_t _processingUnits; //ProcessingUnits able to interact with this MemoryPlace

protected:
    AddressSpace * _addressSpace;
    int _index;	
	
public:
	MemoryPlace(int index, AddressSpace * addressSpace = nullptr)
        : _index(index), _addressSpace(addressSpace)
	{}
    
    virtual ~MemoryPlace() {}
	const size_t getPUCount(void){ return _processingUnits.size(); }
	const ComputePlace* getPU(int index){ return _processingUnits[index]; }
	inline int getIndex(void){ return _index; } 
	void addPU(ComputePlace* pu);
	const std::vector<int>* getPUIndexes();
	const std::vector<ComputePlace*>* getPUs();
};

#endif //MEMORY_PLACE_HPP
