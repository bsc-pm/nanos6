#ifndef ADDRESS_SPACE_HPP
#define ADDRESS_SPACE_HPP

#include <vector>
#include <map>

class MemoryPlace;

class AddressSpace {
private:
	typedef std::map<int, MemoryPlace*> memoryPlaces_t;
	memoryPlaces_t _memoryPlaces; // MemoryPlaces within this AddressSpace 

protected:
    unsigned int _index;	
	
public:
	AddressSpace(unsigned int index)
        : _index(index)
	{}
    
    virtual ~AddressSpace() {}
	const size_t getMemoryPlacesCount(void){ return _memoryPlaces.size(); }
	const MemoryPlace* getMemoryPlace(unsigned int index){ return _memoryPlaces[index]; }
	void addMemoryPlace(MemoryPlace* memoryPlace);
	const std::vector<unsigned int>* getMemoryPlacesIndexes();
	const std::vector<MemoryPlace*>* getMemoryPlaces();
};

#endif //ADDRESS_SPACE_HPP
