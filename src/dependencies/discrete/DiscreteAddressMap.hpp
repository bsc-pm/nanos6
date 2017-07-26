/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DISCRETE_ADDRESS_MAP_HPP
#define DISCRETE_ADDRESS_MAP_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include "DataAccessRange.hpp"



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
	
	DiscreteAddressMapNode(DataAccessRange accessRange)
		: _mapLinks(), _contents(accessRange)
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
	
	DataAccessRange const &getAccessRange() const
	{
		return _contents.getAccessRange();
	}
	
};


template <typename ContentType>
struct KeyOfDiscreteAddressMapNodeArtifact
{
	typedef DataAccessRange type;
	
	inline const type & operator()(DiscreteAddressMapNode<ContentType> const &node) const
	{
		return node._contents.getAccessRange();
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
	
	ContentType &operator[](DataAccessRange accessRange)
	{
		auto it = _map.find(accessRange);
		if (it != _map.end()) {
			return it->_contents;
		} else {
			DiscreteAddressMapNode<ContentType> *newNode = new DiscreteAddressMapNode<ContentType>(accessRange);
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
	
	const_iterator find(DataAccessRange const &range) const
	{
		return _map.find(range);
	}
	
	iterator find(DataAccessRange const &range)
	{
		return _map.find(range);
	}
	
	iterator insert(ContentType &&content)
	{
		assert(!exists(content.getAccessRange(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
		DiscreteAddressMapNode<ContentType> *node = new DiscreteAddressMapNode<ContentType>(std::move(content));
		std::pair<typename map_t::iterator, bool> insertReturnValue = _map.insert(*node);
		return insertReturnValue.first;
	}
	
	iterator insert(ContentType const &content)
	{
		assert(!exists(content.getAccessRange(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
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
	
	
	//! \brief Pass all elements that intersect a given range through a lambda
	//! 
	//! \param[in] range the range to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting the range and that returns a boolean, that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processIntersecting(DataAccessRange const &range, ProcessorType processor)
	{
		auto it = _map.find(range);
		if (it != _map.end()) {
			return processor(it);
		}
		
		return true;
	}
	
	
	//! \brief Pass all elements that intersect a given range through a lambda and any missing subranges through another lambda
	//! 
	//! \param[in] range the range to explore
	//! \param[in] intersectingProcessor a lambda that receives an iterator to each element intersecting the range and that returns a boolean equal to false to stop the traversal
	//! \param[in] missingProcessor a lambda that receives each missing subrange as a DataAccessRange  and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissing(DataAccessRange const &range, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor)
	{
		auto it = _map.find(range);
		if (it != _map.end()) {
			return intersectingProcessor(it);
		} else {
			return missingProcessor(range);
		}
	}
	
	//! \brief Traverse a range of elements to check if there is an element that matches a given condition
	//! 
	//! \param[in] range the range to explore
	//! \param[in] condition a lambda that receives an iterator to each element intersecting the range and that returns the result of evaluating the condition
	//! 
	//! \returns true if the condition evaluated to true for any element
	template <typename PredicateType>
	bool exists(DataAccessRange const &range, PredicateType condition)
	{
		auto it = _map.find(range);
		if (it != _map.end()) {
			return condition(it);
		}
		
		return false;
	}
	
	
	//! \brief Check if there is any element in a given range
	//! 
	//! \param[in] range the range to explore
	//! 
	//! \returns true if there was at least one element at least partially in the range
	bool contains(DataAccessRange const &range)
	{
		return (_map.find(range) != _map.end());
	}
	
	
	iterator fragmentByIntersection(iterator position, DataAccessRange const &range, bool removeIntersection)
	{
		assert(position != _map.end());
		assert(position == _map.find(range));
		
		if (removeIntersection) {
			return _map.end();
		} else {
			return position;
		}
	}
};


#endif // DISCRETE_ADDRESS_MAP_HPP
