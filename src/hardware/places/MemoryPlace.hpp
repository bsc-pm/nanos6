#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

#include <vector>
#include <map>

#include "HardwarePlace.hpp"
#include "ComputePlace.hpp"

class MemoryPlace : public HardwarePlace{
private:
	typedef std::map<int, ComputePlace*> processors_t; // no guarantee that cpus are in order in a node, map for easy access
 
	size_t _ncpus;
	processors_t _processors;

	friend class Machine;
	friend class Loader;
	
	void _addPU(ComputePlace* pu){
		_processors[pu->_index] = pu;
		_ncpus++;
	}

	
	MemoryPlace(int index) // TODO: array of CPUs as parameter or map as parameter 
		: HardwarePlace(index, nullptr)
	{ 
	}

	
public:
	
	const size_t getCPUCount(void){
		return _ncpus;
	}

	const ComputePlace* getCPU(int os_index){
		return _processors[os_index];
	}

	const std::vector<int>* getCPUIndexes(){
	
		std::vector<int>* indexes = new std::vector<int>();
	
		for(processors_t::iterator it = _processors.begin(); it != _processors.end(); ++it){
			indexes->push_back(it->first);
		}
	
		return indexes;
	}

	const std::vector<ComputePlace*>* getCPUs(){
		std::vector<ComputePlace*>* cpus = new std::vector<ComputePlace*>();
		
		for(processors_t::iterator it = _processors.begin(); it != _processors.end(); ++it){
			cpus->push_back(it->second);
		}
		
		return cpus;
	}
};

#endif //MEMORY_PLACE_HPP
