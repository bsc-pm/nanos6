#ifndef DATA_TRANSFER_HPP
#define DATA_TRANSFER_HPP

#include <functional>

#include "hardware/places/MemoryPlace.hpp"

#include <DataAccessRegion.hpp>

class MemoryPlace;

class DataTransfer {
public:
	//! The region that is being transfered
	DataAccessRegion _region;
	
	//! Source memory place
	MemoryPlace const *_source;
	
	//! Target memory place
	MemoryPlace const *_target;

private:
	typedef std::function<void ()> data_transfer_callback_t;
	
	//! The callback that we will invoke when the DataTransfer completes
	data_transfer_callback_t _callback;
	
	//! Flag indicating DataTransfer completion
	bool _completed;
	
public:
	DataTransfer(DataAccessRegion const &region, MemoryPlace const *source,
		MemoryPlace const *target)
		: _region(region), _source(source), _target(target),
		_callback(), _completed(false)
	{
	}
	
	virtual ~DataTransfer()
	{
	}
	
	//! \brief Set the callback for the DataTransfer
	//!
	//! \param[in] callback is the completion callback
	inline void setCompletionCallback(data_transfer_callback_t callback)
	{
		_callback = callback;
	}
	
	//! \brief Mark the DataTransfer as completed
	//!
	//! If there is a valid callback assigned to the DataTransfer it will
	//! be invoked
	inline void markAsCompleted()
	{
		if (_callback) {
			_callback();
		}
		
		_completed = true;
	}
	
	//! \brief Check if the DataTransfer is completed
	inline bool isCompleted() const
	{
		return _completed;
	}
	
	friend std::ostream& operator<<(std::ostream &out,
			const DataTransfer &dt)
	{
		out << "DataTransfer from: " <<
			dt._source->getIndex() << " to: " <<
			dt._target->getIndex() << " region:" <<
			dt._region;
		return out;
	}
};

#endif /* DATA_TRANSFER_HPP */
