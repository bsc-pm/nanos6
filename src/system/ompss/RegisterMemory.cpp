/*#include "api/nanos6_rt_interface.h"
#include "memory/Directory.hpp"

#include <numa.h>

bool nanos_register_region(void *address, unsigned long size, int mode, nodemask_t nodemask){

	bitmask bm = numa_allocate_nodemask(); 
	copy_nodemask_to_bitmask(&nodemask, &bm);
	
	int nNodes = numa_bitmask_weight(bm);
	vector<int> indexes = Machine.getNodeIndexes();
	MemoryLocation** locations = new MemoryLocation*[nNodes];
	bool interleaved = false;
	
	if(mode == MPOL_INTERLEAVED){
		interleaved = true;
		int i = 0;
		// store all nodes in which it is allocated
		for(std::vector<int>::iterator index = indexes.begin(); index != indexes.end(); ++index){
			if( numa_bitmask_isbitset(bm, *index) ){
				locations[i] = Machine.getNode(*index);
				++i;	
			}
		}
		
		// insert interleaved and present
		Directory.insert(address, size, true, nNodes, true, locations);
	} else if(mode == MPOL_BIND && nNodes == 1){
		
		// find the node in which it is bound
		for(set::vector<int>::iterator index = indexes.begin(); index != indexes.end(); ++index){
			if( numa_bitmask_isbitset(bm, *index) ){
				locations[0] = Machine.getNode(*index);
				break;
			}
		}

		// insert not interleaved and present
		Directory.insert(address, size, false, 1, true, locations);
	} else {
		
		// insert with no data (all default)
		Directory.insert(address, size);
	}
	
	return true;
			
}
*/
