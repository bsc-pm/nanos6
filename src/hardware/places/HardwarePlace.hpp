#ifndef HARDWARE_PLACE_HPP
#define HARDWARE_PLACE_HPP


//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class HardwarePlace {
protected:
	HardwarePlace *_parent;
	
public:
	void *_schedulerData;
	
	HardwarePlace(HardwarePlace *parent = nullptr)
		: _parent(parent), _schedulerData(nullptr)
	{
	}
	
	virtual ~HardwarePlace()
	{
	}
	
	
};


#endif // HARDWARE_PLACE_HPP
