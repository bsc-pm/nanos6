#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include "MemoryRegion.hpp"
#include "dependencies/linear-regions-unfragmented/LinearRegionMap.hpp" 
#include "dependencies/linear-regions-unfragmented/LinearRegionMapImplementation.hpp"

class DirectoryNode {
	
friend class LinearRegionMap<DirectoryNode>;

private:
	MemoryRegion* _region;
	DataAccessRange _accessRange;
public:

	DirectoryNode(MemoryRegion* region)
	: _region( region ),
	  _accessRange( region->getAccessRange() )
	{

	}

	~DirectoryNode(){
		delete _region;
	}

	MemoryRegion* getRegion(){
		return _region;
	}

	DataAccessRange const &getAccessRange() const
        {
                return _accessRange;
        }

        DataAccessRange &getAccessRange()
        {
                return _accessRange;
        }

};

class Directory : LinearRegionMap<DirectoryNode> {
private:
	
	static Directory *_instance;
	
	Directory(){
		
	}

public:		
	
	// singleton methods
	static void init(){
		_instance = new Directory();
	}

	static void destroy(){
		delete _instance;
	}

	static bool addRegion(void *addr, size_t size, bool interleaved = false, int nLocations = 0, bool present = false, MemoryPlace **locations = nullptr){
        // create region object
        	MemoryRegion *region = new MemoryRegion(addr, size, interleaved, nLocations, present, locations);

        	// merge with intersecting regions when not interleaved and present
        	if(!interleaved && present){

                	_instance->processIntersecting(
                        	region->_range,
                        	[&](Directory::iterator position) -> bool {
                                	region->merge(position->getRegion());
                                	_instance->erase(position);
                                	return true;
                        	}
                	);
        	}
	
        	// insert to directory
        	_instance->insert(DirectoryNode(region));
        	return true;
	}
	
};



#endif //DIRECTORY_HPP
