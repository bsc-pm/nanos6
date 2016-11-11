#ifndef REGION_SET_HPP
#define REGION_SET_HPP

#include <IntrusiveLinearRegionMap.hpp>
#include <IntrusiveLinearRegionMapImplementation.hpp>

#include "MemoryPageObject.hpp"
#include "dependencies/linear-regions/DataAccessRange.hpp"

class MemoryPageSet: public IntrusiveLinearRegionMap<MemoryPageObject, boost::intrusive::function_hook< MemoryPageObjectLinkingArtifacts > >{

private:
	
	typedef IntrusiveLinearRegionMap<MemoryPageObject, boost::intrusive::function_hook< MemoryPageObjectLinkingArtifacts > > BaseType; 
	
public:
	
	MemoryPageSet();
	iterator find(void *address);
	void insert(DataAccessRange range);	
	
};

#endif //REGION_SET_HPP
