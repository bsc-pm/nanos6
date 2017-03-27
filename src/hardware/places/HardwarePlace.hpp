#ifndef HARDWARE_PLACE_HPP
#define HARDWARE_PLACE_HPP

#include <InstrumentHardwarePlaceId.hpp>

#include "CPUDependencyData.hpp"


//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class HardwarePlace {
protected:
	HardwarePlace *_parent;
	Instrument::hardware_place_id_t _instrumentationId;
	
	CPUDependencyData _dependencyData;
	
public:
	void *_schedulerData;
	
	HardwarePlace(HardwarePlace *parent = nullptr)
		: _parent(parent), _schedulerData(nullptr)
	{
	}
	
	virtual ~HardwarePlace()
	{
	}
	
	void setInstrumentationId(Instrument::hardware_place_id_t const &instrumentationId)
	{
		_instrumentationId = instrumentationId;
	}
	
	Instrument::hardware_place_id_t const &getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	CPUDependencyData &getDependencyData()
	{
		return _dependencyData;
	}
	
};


#endif // HARDWARE_PLACE_HPP
