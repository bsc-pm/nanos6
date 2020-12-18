/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_TRACKING_HPP
#define DATA_ACCESS_TRACKING_HPP

#include "DataAccess.hpp"
#include "dependencies/DataTrackingSupport.hpp"

struct DataAccessTracking : public DataAccess {
private:
	//! Data tracking
	DataTrackingSupport::DataTrackingInfo _trackingInfo;

public:
	DataAccessTracking(DataAccessType type, Task *originator, void *address, size_t length, bool weak) :
		DataAccess(type, originator, address, length, weak), _trackingInfo()
	{
	}

	virtual inline DataTrackingSupport::DataTrackingInfo *getTrackingInfo()
	{
		return &_trackingInfo;
	}

	virtual void updateTrackingInfo(DataTrackingSupport::location_t location, DataTrackingSupport::timestamp_t timeL2, DataTrackingSupport::timestamp_t timeL3)
	{
		assert(DataTrackingSupport::isTrackingEnabled());

		// If there is a location, it must have a valid timeL2
		assert(!((location != DataTrackingSupport::UNKNOWN_LOCATION) && (timeL2 == DataTrackingSupport::NOT_PRESENT)));
		// If there is no location, it cannot have a valid timeL2
		assert(!((location == DataTrackingSupport::UNKNOWN_LOCATION) && (timeL2 != DataTrackingSupport::NOT_PRESENT)));
		// If there is no location, it cannot have a valid timeL3
		assert(!((location == DataTrackingSupport::UNKNOWN_LOCATION) && (timeL3 != DataTrackingSupport::NOT_PRESENT)));
		assert(location == DataTrackingSupport::UNKNOWN_LOCATION || location < (int) HardwareInfo::getNumL2Cache());

		_trackingInfo.setInfo(location, timeL2, timeL3);
	}
};


#endif // DATA_ACCESS_TRACKING_HPP
