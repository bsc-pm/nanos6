#ifndef DATA_ACCESS_RANGE_HPP
#define DATA_ACCESS_RANGE_HPP


#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ostream>
#include <utility>


class DataAccessRange;
inline std::ostream & operator<<(std::ostream &o, DataAccessRange const &range);


class DataAccessRange {
private:
	//! The starting address of the data access
	void *_startAddress;
	
	//! The length of the accesses
	size_t _length;
	
	struct FragmentBoundaries {
		char *_firstStart;
		char *_firstEnd;
		char *_secondStart;
		char *_secondEnd;
		
		FragmentBoundaries(DataAccessRange const &first, DataAccessRange const &second)
		{
			_firstStart = (char *) first._startAddress;
			_firstEnd = _firstStart + first._length;
			_secondStart = (char *) second._startAddress;
			_secondEnd = _secondStart + second._length;
		}
	};
	
	
public:
	DataAccessRange(void *startAddress, size_t length)
		: _startAddress(startAddress), _length(length)
	{
	}
	
	DataAccessRange(void *startAddress, void *endAddress)
	: _startAddress(startAddress)
	{
		char *start = (char *)startAddress;
		char *end = (char *)endAddress;
		_length = (size_t) (end - start);
	}
	
	DataAccessRange()
		: _startAddress(0), _length(0)
	{
	}
	
	bool empty() const
	{
		return (_startAddress == nullptr) && (_length == 0);
	}
	
	bool operator==(DataAccessRange const &other) const
	{
		return (_startAddress == other._startAddress) && (_length == other._length);
	}
	
	bool operator!=(DataAccessRange const &other) const
	{
		return (_startAddress != other._startAddress) || (_length != other._length);
	}
	
	
	void *getStartAddress() const
	{
		return _startAddress;
	}
	
	void * const &getStartAddressConstRef() const
	{
		return _startAddress;
	}
	
	void *getEndAddress() const
	{
		char *start = (char *) _startAddress;
		char *end = ((char *) start) + _length;
		
		return end;
	}
	
	size_t getSize() const
	{
		return _length;
	}
	
	
	std::pair<void *, void *> getBounds() const
	{
		char *start = (char *) _startAddress;
		char *end = ((char *) start) + _length - 1;
		
		return std::pair<void *, void *>(start, end);
	}
	
	
	//! \brief Returns the intersection or an empty DataAccessRange if there is none
	DataAccessRange intersect(DataAccessRange const &other) const
	{
		FragmentBoundaries boundaries(*this, other);
		
		char *start = std::max(boundaries._firstStart, boundaries._secondStart);
		char *end = std::min(boundaries._firstEnd, boundaries._secondEnd);
		
		if (start < end) {
			return DataAccessRange(start, end);
		} else {
			return DataAccessRange();
		}
	}
	
	
	bool fullyContainedIn(DataAccessRange const &other) const
	{
		return intersect(other) == *this;
	}
	
	
	template <typename ThisOnlyProcessorType, typename IntersectingProcessorType, typename OtherOnlyProcessorType>
	void processIntersectingFragments(
		DataAccessRange const &fragmeterRange,
		ThisOnlyProcessorType thisOnlyProcessor,
		IntersectingProcessorType intersectingProcessor,
		OtherOnlyProcessorType otherOnlyProcessor
	) {
		FragmentBoundaries boundaries(*this, fragmeterRange);
		
		char *intersectionStart = std::max(boundaries._firstStart, boundaries._secondStart);
		char *intersectionEnd = std::min(boundaries._firstEnd, boundaries._secondEnd);
		
		// There must be an intersection
		assert(intersectionStart < intersectionEnd);
		
		// Intersection
		DataAccessRange intersection(intersectionStart, intersectionEnd);
		intersectingProcessor(intersection);
		
		// Left of intersection
		if (boundaries._firstStart < intersectionStart) {
			DataAccessRange leftOfIntersection(boundaries._firstStart, intersectionStart);
			thisOnlyProcessor(leftOfIntersection);
		} else if (boundaries._secondStart < intersectionStart) {
			DataAccessRange leftOfIntersection(boundaries._secondStart, intersectionStart);
			otherOnlyProcessor(leftOfIntersection);
		}
		
		// Right of intersection
		if (intersectionEnd < boundaries._firstEnd) {
			DataAccessRange rightOfIntersection(intersectionEnd, boundaries._firstEnd);
			thisOnlyProcessor(rightOfIntersection);
		} else if (intersectionEnd < boundaries._secondEnd) {
			DataAccessRange rightOfIntersection(intersectionEnd, boundaries._secondEnd);
			otherOnlyProcessor(rightOfIntersection);
		}
	}
	
	friend std::ostream & ::operator<<(std::ostream &o, DataAccessRange const &range);
};


inline std::ostream & operator<<(std::ostream &o, const DataAccessRange& range)
{
	return o << range._startAddress << ":" << range._length;
}


#endif // DATA_ACCESS_RANGE_HPP
