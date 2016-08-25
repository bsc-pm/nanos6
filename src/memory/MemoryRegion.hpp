#ifndef REGION_HPP
#define REGION_HPP

#include "../dependencies/linear-regions-unfragmented/DataAccessRange.hpp"
#include "../hardware/places/MemoryPlace.hpp"

class MemoryRegion {
	
friend class Directory;

private:
	void *_address; //< start address of the region
	size_t _size; //< length of the region
	bool _interleaved; //< true if data interleaved between nodes
	size_t _nLocations; //< number of nodes in which it is interleaved
	bool _present; //< true if data is already located	

	DataAccessRange _range;

	MemoryPlace **_location; //< memory nodes of the region
public:

	MemoryRegion( void *address, size_t size, bool interleaved = false, size_t nLocations = 1, bool present = false, MemoryPlace **location = nullptr )
		: _address( address ),
		_size( size ),
		_range( address, size ),
		_interleaved( interleaved ),
		_nLocations( nLocations ),
		_present( present ),
		_location( location )
	{
		// Consider changing the address to the page start address
	}

	void merge( MemoryRegion *other ); // merge two regions into this
	int locate( void ); // find the pages of this memory region
	
	DataAccessRange const &getAccessRange() const
        {
                return _range;
        }

        DataAccessRange &getAccessRange()
        {
                return _range;
        }

};

#endif //REGION_HPP
