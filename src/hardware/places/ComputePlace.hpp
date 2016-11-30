#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP

#include <map>
#include <vector>

class MemoryPlace;

//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class ComputePlace {
private:
	typedef std::map<int, MemoryPlace*> memoryPlaces_t;
    memoryPlaces_t _memoryPlaces; // Accessible MemoryPlaces from this ComputePlace 

protected:
    //ComputePlace * _parent;
    int _index;	

public:
    void *_schedulerData;
    
	ComputePlace(int index/*, ComputePlace *parent = nullptr*/)
		: _index(index)/*, _parent(parent)*/, _schedulerData(nullptr)
	{}
	
	virtual ~ComputePlace() {}
	size_t getMemoryPlacesCount(void) const { return _memoryPlaces.size(); }
	MemoryPlace* getMemoryPlace(int index) { 
        memoryPlaces_t::iterator it = _memoryPlaces.find(index);
        if(it != _memoryPlaces.end()) {
            return it->second;
        }
        return nullptr;
    }
	inline int getIndex(void) const{ return _index; } 
	void addMemoryPlace(MemoryPlace* mem);
	std::vector<int> getMemoryPlacesIndexes();
	std::vector<MemoryPlace*> getMemoryPlaces();
	
};

#endif //COMPUTE_PLACE_HPP
