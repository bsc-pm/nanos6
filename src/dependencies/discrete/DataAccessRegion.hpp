/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGION_HPP
#define DATA_ACCESS_REGION_HPP


#include <cstddef>
#include <ostream>


class DataAccessRegion;
inline std::ostream & operator<<(std::ostream &o, DataAccessRegion const &region);


class DataAccessRegion {
private:
	//! The starting address of the data access
	void *_startAddress;
	
	//! For now we are not considering the length of the accesses
	
public:
	DataAccessRegion(void *startAddress, __attribute__((unused)) size_t length)
		: _startAddress(startAddress)
	{
	}
	
	DataAccessRegion()
		: _startAddress(0)
	{
	}
	
	bool operator<(DataAccessRegion const &other) const
	{
		return _startAddress < other._startAddress;
	}
	
	bool operator==(DataAccessRegion const &other) const
	{
		return _startAddress == other._startAddress;
	}
	
	bool operator!=(DataAccessRegion const &other) const
	{
		return _startAddress != other._startAddress;
	}
	
	bool empty() const
	{
		return (_startAddress == nullptr);
	}
	
	void *getStartAddress() const
	{
		return _startAddress;
	}
	
	void * const &getStartAddressConstRef() const
	{
		return _startAddress;
	}
	
	//! \brief Returns the intersection or an empty DataAccessRegion if there is none
	DataAccessRegion intersect(DataAccessRegion const &other) const
	{
		if (other == *this) {
			return other;
		} else {
			return DataAccessRegion();
		}
	}
	
	
	bool fullyContainedIn(DataAccessRegion const &other) const
	{
		return other == *this;
	}
	
	
	template <typename ThisOnlyProcessorType, typename IntersectingProcessorType, typename OtherOnlyProcessorType>
	void processIntersectingFragments(
		DataAccessRegion const &fragmeterRegion,
		ThisOnlyProcessorType thisOnlyProcessor,
		IntersectingProcessorType intersectingProcessor,
		OtherOnlyProcessorType otherOnlyProcessor
	) {
		if (fragmeterRegion == *this) {
			intersectingProcessor(fragmeterRegion);
		} else {
			thisOnlyProcessor(*this);
			otherOnlyProcessor(fragmeterRegion);
		}
	}
	
	
	friend std::ostream & ::operator<<(std::ostream &o, DataAccessRegion const &region);
};


inline std::ostream & operator<<(std::ostream &o, const DataAccessRegion& region)
{
	return o << region._startAddress;
}




#endif // DATA_ACCESS_REGION_HPP
