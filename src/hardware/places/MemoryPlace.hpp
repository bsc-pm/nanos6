#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

class MemoryPlace : public HardwarePlace{
private:
	typedef std::map<int, ComputePlace*> processors_t; // no guarantee that cpus are in order in a node, map for easy access
 
	size_t _ncpus;
	processors_t *_processors;

	
	void _addPU(ComputePlace* pu){
		processors_t[pu->os_index] = pu;
		ncpus++;
	}

	
	MemoryPlace(int index) // TODO: array of CPUs as parameter or map as parameter 
		: HardwarePlace(index, nullptr)
	{ 
	}
public:

};

#endif //MEMORY_PLACE_HPP
