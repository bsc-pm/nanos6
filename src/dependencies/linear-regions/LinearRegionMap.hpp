/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef LINEAR_REGION_MAP_HPP
#define LINEAR_REGION_MAP_HPP

#include <utility>

#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/intrusive/parent_from_member.hpp>
#include <boost/version.hpp>
#include "DataAccessRange.hpp"


namespace LinearRegionMapInternals {
	template <typename ContentType>
	struct Node {
		#if NDEBUG
			typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
		#else
			typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
		#endif
		typedef boost::intrusive::avl_set_member_hook<link_mode_t> links_t;
		
		links_t _mapLinks;
		ContentType _contents;
		
		
		Node() = delete;
		Node(Node const &other) = delete;
		Node(Node &&other) = delete;
		
		Node(DataAccessRange accessRange)
		: _mapLinks(), _contents(accessRange)
		{
		}
		
		Node(ContentType &&contents)
			: _mapLinks(), _contents(std::move(contents))
		{
		}
		
		Node(ContentType const &contents)
		: _mapLinks(), _contents(contents)
		{
		}
		
		DataAccessRange const &getAccessRange() const
		{
			return _contents.getAccessRange();
		}
		
		DataAccessRange &getAccessRange()
		{
			return _contents.getAccessRange();
		}
	};
	
	
	struct address_t
	{
		void *_address;
		
		inline address_t(void *address)
			: _address(address)
		{
		}
		
		inline bool operator<(address_t const &other) const
		{
			return _address < other._address;
		}
	};
	
	
	template <typename ContentType>
	struct KeyOfNodeArtifact
	{
#if BOOST_VERSION >= 106200
		typedef address_t type;
		
		type operator()(Node<ContentType> const &node)
		{ 
			return node.getAccessRange().getStartAddressConstRef();
		}
#else
		typedef void *type;
		
		type const &operator()(Node<ContentType> const &node)
		{ 
			return node.getAccessRange().getStartAddressConstRef();
		}
#endif
	};
}



template <typename ContentType>
class LinearRegionMap {
private:
	typedef boost::intrusive::avl_set<
		LinearRegionMapInternals::Node<ContentType>,
		boost::intrusive::key_of_value<LinearRegionMapInternals::KeyOfNodeArtifact<ContentType>>,
		boost::intrusive::member_hook<
			LinearRegionMapInternals::Node<ContentType>,
			typename LinearRegionMapInternals::Node<ContentType>::links_t,
			&LinearRegionMapInternals::Node<ContentType>::_mapLinks
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
	
	LinearRegionMap()
		: _map()
	{
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
		return _map.find(range.getStartAddress());
	}
	
	iterator find(DataAccessRange const &range)
	{
		return _map.find(range.getStartAddress());
	}
	
	iterator insert(ContentType &&content)
	{
		assert(!exists(content.getAccessRange(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
		LinearRegionMapInternals::Node<ContentType> *node = new LinearRegionMapInternals::Node<ContentType>(std::move(content));
		std::pair<typename map_t::iterator, bool> insertReturnValue = _map.insert(*node);
		return insertReturnValue.first;
	}
	
	iterator insert(ContentType const &content)
	{
		assert(!exists(content.getAccessRange(), [&](__attribute__((unused)) iterator position) -> bool { return true; }));
		
		LinearRegionMapInternals::Node<ContentType> *node = new LinearRegionMapInternals::Node<ContentType>(content);
		std::pair<typename map_t::iterator, bool> insertReturnValue = _map.insert(*node);
		return insertReturnValue.first;
	}
	
	iterator erase(iterator position)
	{
		typename map_t::iterator it = position;
		LinearRegionMapInternals::Node<ContentType> *node = &(*it);
		iterator result = _map.erase(position);
		delete node;
		return result;
	}
	
	void moved(ContentType *content)
	{
		typedef LinearRegionMapInternals::Node<ContentType> node_t;
		
		node_t *node = boost::intrusive::get_parent_from_member<node_t>(content, &node_t::_contents);
		typename map_t::iterator it = _map.iterator_to(*node);
		
		_map.erase(it);
		_map.insert(*node);
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
	bool processAll(ProcessorType processor);
	
	//! \brief Pass all elements that intersect a given range through a lambda
	//! 
	//! \param[in] range the range to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting the range and that returns a boolean, that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processIntersecting(DataAccessRange const &range, ProcessorType processor);
	
	//! \brief Pass all elements that intersect a given range through a lambda and any missing subranges through another lambda
	//! 
	//! \param[in] range the range to explore
	//! \param[in] intersectingProcessor a lambda that receives an iterator to each element intersecting the range and that returns a boolean equal to false to stop the traversal
	//! \param[in] missingProcessor a lambda that receives each missing subrange as a DataAccessRange  and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissing(DataAccessRange const &range, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor);
	
	//! \brief Traverse a range of elements to check if there is an element that matches a given condition
	//! 
	//! \param[in] range the range to explore
	//! \param[in] condition a lambda that receives an iterator to each element intersecting the range and that returns the result of evaluating the condition
	//! 
	//! \returns true if the condition evaluated to true for any element
	template <typename PredicateType>
	bool exists(DataAccessRange const &range, PredicateType condition);
	
	
	//! \brief Check if there is any element in a given range
	//! 
	//! \param[in] range the range to explore
	//! 
	//! \returns true if there was at least one element at least partially in the range
	bool contains(DataAccessRange const &range);
	
	
	iterator fragmentByIntersection(iterator position, DataAccessRange const &range, bool removeIntersection);
	
};



#endif // LINEAR_REGION_MAP_HPP
