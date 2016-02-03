#ifndef FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
#define FIXED_ADDRESS_DATA_ACCESS_MAP_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include "DataAccessRange.hpp"
#include "DataAccessSequence.hpp"
#include "lowlevel/SpinLock.hpp"


class FixedAddressDataAccessMap {
public:
	typedef void *address_t;
	
	
private:
	struct Node {
		#if NDEBUG
			typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
		#else
			typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
		#endif
		typedef boost::intrusive::avl_set_member_hook<link_mode_t> links_t;
		
		links_t _mapLinks;
		SpinLock _lock;
		DataAccessSequence _accessSequence;
		
		Node()
			: _mapLinks(), _lock(), _accessSequence(&_lock)
		{
		}
		
		Node(DataAccessRange accessRange)
		: _mapLinks(), _lock(), _accessSequence(accessRange, &_lock)
		{
		}
	};
	
	
	struct KeyOfNodeArtifact
	{
		typedef DataAccessRange type;
		
		const type & operator()(Node const &node) const
		{ 
			return node._accessSequence._accessRange;
		}
	};
	
	
	typedef boost::intrusive::avl_set< Node, boost::intrusive::key_of_value<KeyOfNodeArtifact>, boost::intrusive::member_hook<Node, Node::links_t, &Node::_mapLinks> > map_t;
	
	
	map_t _map;
	
	
public:
	FixedAddressDataAccessMap()
		: _map()
	{
	}
	
	DataAccessSequence &operator[](DataAccessRange accessRange)
	{
		auto it = _map.find(accessRange);
		if (it != _map.end()) {
			return it->_accessSequence;
		} else {
			Node *newNode = new Node(accessRange);
			_map.insert(*newNode); // This operation does actually take the pointer
			return newNode->_accessSequence;
		}
	}
	
};


#endif // FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
