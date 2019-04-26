/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_DMALLOC_HPP
#define MESSAGE_DMALLOC_HPP

#include "Message.hpp"

#include <nanos6/cluster.h>

class MessageDmalloc : public Message {
	struct DmallocMessageContent {
		//! size in bytes of the requested allocation
		size_t _allocationSize;
		
		//! distribution policy for the region
		nanos6_data_distribution_t _policy;
		
		//! number of dimensions for distribution
		size_t _nrDim;
		
		//! dimensions of the distribution
		size_t _dimensions[];
	};
	
	//! \brief pointer to the message payload
	DmallocMessageContent *_content;
	
public:
	MessageDmalloc(const ClusterNode *from, size_t num_dimensions);
	
	MessageDmalloc(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<DmallocMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	//! \brief Set the allocation size
	inline void setAllocationSize(size_t size)
	{
		_content->_allocationSize = size;
	}
	
	//! \brief Get the allocation size
	inline size_t getAllocationSize() const
	{
		return _content->_allocationSize;
	}
	
	//! \brief Set distribution policy
	inline void setDistributionPolicy(nanos6_data_distribution_t policy)
	{
		_content->_policy = policy;
	}
	
	//! \brief Get distribution policy
	inline nanos6_data_distribution_t getDistributionPolicy() const
	{
		return _content->_policy;
	}
	
	//! \brief Get policy dimensions size
	inline size_t getDimensionsSize() const
	{
		return _content->_nrDim;
	}
	
	//! \brief Set policy dimensions
	inline void setDimensions(size_t *dimensions)
	{
		memcpy(_content->_dimensions, dimensions,
			sizeof(size_t) * _content->_nrDim);
	}
	
	//! \brief Get policy dimensions
	inline size_t *getDimensions() const
	{
		if (_content->_nrDim == 0) {
			return nullptr;
		}
		
		return _content->_dimensions;
	}
	
	//! \brief write to a stream a description of the Message
	inline void toString(std::ostream &where) const
	{
		where << "Distributed allocation of "
			<< _content->_allocationSize
			<< " bytes.";
	}
};

#endif /* MESSAGE_DMALLOC_HPP */
