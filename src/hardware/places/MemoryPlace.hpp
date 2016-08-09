#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

class MemoryPlace : public HardwarePlace{
private:
	typedef std::map<int, ComputePlace*> processors_t; // no guarantee that cpus are in order in a node, map for easy access
 
	size_t _ncpus;
	processors_t _processors;

	
	void _addPU(ComputePlace* pu){
		processors[pu->os_index] = pu;
		ncpus++;
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
		return _processorsi[os_index];
	}

	const vector<int>* getCPUIndexes(){
	
		std::vector<int>* indexes = new std::vector<int>();
	
		for(processors_t::iterator it = _processors.begin(); it != _processors.end(); ++it){
			indexes.push_back(it->first);
		}
	
		return indexes;
	}

	const vector<ComputePlace*>* getCPUs(){
		std::vector<ComputePlace*>* cpus = new std::vector<int>();
		
		for(processors_t::iterator it = _processors.begin(); it != _processors.end(); ++it){
			cpus.push_back(it->second);
		}
		
		return cpus;
	}
};

#endif //MEMORY_PLACE_HPP
