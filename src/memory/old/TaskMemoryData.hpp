#ifndef TASK_MEM_DATA
#define TASK_MEM_DATA

#include <vector>
#include <cstddef>

struct TaskMemoryData{
	enum class State {READY, COPY_READY, COPYING, COPIED} //Need to define states

	State state;
	std::vector<void *> bases;
	std::vector<size_t> sizes;
	std::vector<int> ids;
	int regions;	
	

	TaskMemoryData() : bases(), sizes(), ids(), regions(0){
			
	}

	void add(void *baseAddress, size_t size, int id){
		bases.push_back(baseAddress);
		sizes.push_back(size);
		ids.push_back(id);
		regions++;
	}
};

typedef struct TaskMemoryData TaskMemoryData;

#endif //TASK_MEM_DATA
