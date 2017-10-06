/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DISCRETE_ADDRESS_MAP_HPP
#define DISCRETE_ADDRESS_MAP_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include "DataAccessRegion.hpp"



template <typename ContentType>
struct DiscreteAddressMapNode {
	#if NDEBUG
	typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
	typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif
	typedef boost::intrusive::avl_set_member_hook<link_mode_t> links_t;
	
	links_t _mapLinks;
	ContentType _contents;
	
	DiscreteAddressMapNode()
		: _mapLinks(), _contents()
	{
	}
	
	DiscreteAddressMapNode(DataAccessRegion accessRegion)
		: _mapLinks(), _contents(accessRegion)
	{
	}
	
	DiscreteAddressMapNode(ContentType &&contents)
	: _mapLinks(), _contents(std::move(contents))
	{
	}
	
	DiscreteAddressMapNode(ContentType const &contents)
	: _mapLinks(), _contents(contents)
	{
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _contents.getAccessRegion();
	}
	
};


template <typename ContentType>
struct KeyOfDiscreteAddressMapNodeArtifact
{
	typedef DataAccessRegion type;
	
	inline const type & operator()(DiscreteAddressMapNode<ContentType> const &node) const
	{
		return node._contents.getAccessRegion();
	}
};



template <typename ContentType>
class DiscreteAddressMap {
private:
	typedef boost::intrusive::avl_set<
		DiscreteAddressMapNode<ContentType>,
		boost::intrusive::key_of_value<KeyOfDiscreteAddressMapNodeArtifact<ContentType>>,
		boost::intrusive::member_hook<
			DiscreteAddressMapNode<ContentType>,
			typename DiscreteAddressMapNode<ContentType>::links_t,
			&DiscreteAddressMapNode<ContentType>::_mapLinks
		>
	> map_t;
	
	map_t _map;
	
public:
	class iterator : public map_t::iterator {
	public:
		typedef ContentType value_type;
		typedef ContentType *pointer;
		typedef ContentType &reference;
		
		
		iterator(typename map_t::iterator it)
		: map_t::iterator(it)
		{
		}
		
		reference operator*()
		{
			return map_t::iterator::operator*()._contents;
		}
		
		pointer operator->()
		{
			return &map_t::iterator::operator*()._contents;
		}
	};
	
	
	class const_iterator : public map_t::const_iterator {
	public:
		typedef ContentType value_type;
		typedef ContentType const *pointer;
		typedef ContentType const &reference;
		
		
		const_iterator(typename map_t::const_iterator const &it)
		: map_t::const_iterator(it)
		{
		}
		
		reference operator*() const
		{
			return map_t::const_iterator::operator*._contents;
		}
		
		pointer operator->() const
		{
			return &map_t::const_iterator::operator*._contents;
		}
	};
	
	
	typedef typename map_t::size_type size_type;
	
	
	DiscreteAddressMap()
		: _map()
	{
	}
	
	ContentType &operator[](DataAccessRegion accessRegion)
	{
		auto it = _map.find(accessRegion);
		if (it != _map.end()) {
			return it->_contents;
		} else {
			DiscreteAddressMapNode<ContentType> *newNode = new DiscreteAddressMapNode<ContentType>(accessRegion);
			_map.insert(*newNode); // This operation does actually take the pointer
			return newNode->_contents;
		}
	}
	
	iterator begin()
	{
		return _map.begin();
	}
	iterator end()
	{
		return _map.end();
	}
	
	const_iterator begin() const
	{
		return _map.begin();
	}
	const_iterator end() const
	{
		return _map.end();
	}
	
	bool empty() const
	{
		return _map.empty();
	}
	size_type size() const
	{
		return _map.size();
	}
	
	const_iterator find(DataAccessRegion const &region) const
	{
		return _map.find(region);
	}
	
	iterator find(DataAccessRegion const &region)
	{
		return _map.find(region);
	}
	
	iterator insert(ContentType &&content)
	{
		assert(!exists(content.getAccessRegion(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
		DiscreteAddressMapNode<ContentType> *node = new DiscreteAddressMapNode<ContentType>(std::move(content));
		std::pair<typename map_t::iterator, bool> insertReturnValue = _map.insert(*node);
		return insertReturnValue.first;
	}
	
	iterator insert(ContentType const &content)
	{
		assert(!exists(content.getAccessRegion(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
		DiscreteAddressMapNode<ContentType> *node = new DiscreteAddressMapNode<ContentType>(content);
		std::pair<typename map_t::iterator, bool> insertReturnValue = _map.insert(*node);
		return insertReturnValue.first;
	}
	
	iterator erase(iterator position)
	{
		typename map_t::iterator it = position;
		DiscreteAddressMapNode<ContentType> *node = &(*it);
		iterator result = _map.erase(position);
		delete node;
		return result;
	}
	
	void clear()
	{
		for (auto it = begin(); it != end(); ) {
			it = erase(it);
		}
	}
	
	
	//! \brief Pass all elements through a lambda
	//! 
	//! \param[in] processor a lambda that receives an iterator to each element and that returns a boolean, that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processAll(ProcessorType processor)
	{
		for (iterator it = _map.begin(); it != _map.end(); ) {
			iterator position = it;
			it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
			
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
		
		return true;
	}
	
	
	//! \brief Pass all elements that intersect a given region through a lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting the region and that returns a boolean, that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processIntersecting(DataAccessRegion const &region, ProcessorType processor)
	{
		auto it = _map.find(region);
		if (it != _map.end()) {
			return processor(it);
		}
		
		return true;
	}
	
	
	//! \brief Pass all elements that intersect a given region through a lambda and any missing subregions through another lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] intersectingProcessor a lambda that receives an iterator to each element intersecting the region and that returns a boolean equal to false to stop the traversal
	//! \param[in] missingProcessor a lambda that receives each missing subregion as a DataAccessRegion  and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissing(DataAccessRegion const &region, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor)
	{
		auto it = _map.find(region);
		if (it != _map.end()) {
			return intersectingProcessor(it);
		} else {
			return missingProcessor(region);
		}
	}
	
	//! \brief Traverse a region of elements to check if there is an element that matches a given condition
	//! 
	//! \param[in] region the region to explore
	//! \param[in] condition a lambda that receives an iterator to each element intersecting the region and that returns the result of evaluating the condition
	//! 
	//! \returns true if the condition evaluated to true for any element
	template <typename PredicateType>
	bool exists(DataAccessRegion const &region, PredicateType condition)
	{
		auto it = _map.find(region);
		if (it != _map.end()) {
			return condition(it);
		}
		
		return false;
	}
	
	
	//! \brief Check if there is any element in a given region
	//! 
	//! \param[in] region the region to explore
	//! 
	//! \returns true if there was at least one element at least partially in the region
	bool contains(DataAccessRegion const &region)
	{
		return (_map.find(region) != _map.end());
	}
	
	
	iterator fragmentByIntersection(iterator position, DataAccessRegion const &region, bool removeIntersection)
	{
		assert(position != _map.end());
		assert(position == _map.find(region));
		
		if (removeIntersection) {
			return _map.end();
		} else {
			return position;
		}
	}
};


#endif // DISCRETE_ADDRESS_MAP_HPP
