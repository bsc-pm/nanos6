#ifndef HARDWARE_PLACE_HPP
#define HARDWARE_PLACE_HPP


//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class HardwarePlace {
protected:
	HardwarePlace *_parent;
        int _index;	
public:
	HardwarePlace(int index, HardwarePlace *parent = nullptr)
		: _parent(parent),
		_index(index)
	{
	}
	
	virtual ~HardwarePlace()
	{
	}

	int getIndex(void){
		return _index;
	} 
	
};


#endif // HARDWARE_PLACE_HPP
