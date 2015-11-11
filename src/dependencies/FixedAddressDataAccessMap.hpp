#ifndef FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
#define FIXED_ADDRESS_DATA_ACCESS_MAP_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include "DataAccessSequence.hpp"


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
		
		address_t _address;
		links_t _mapLinks;
		DataAccessSequence _accessSequence;
		
		Node()
			: _address(0), _mapLinks(), _accessSequence()
		{
		}
		
		Node(address_t address)
			: _address(address), _mapLinks(), _accessSequence()
		{
		}
	};
	
	
	struct KeyOfNodeArtifact
	{
		typedef address_t type;
		
		const type & operator()(Node const &node) const
		{ 
			return node._address;
		}
	};
	
	
	typedef boost::intrusive::avl_set< Node, boost::intrusive::key_of_value<KeyOfNodeArtifact>, boost::intrusive::member_hook<Node, Node::links_t, &Node::_mapLinks> > map_t;
	
	
	map_t _map;
	
	
public:
	FixedAddressDataAccessMap()
		: _map()
	{
	}
	
	DataAccessSequence &operator[](address_t address)
	{
		auto it = _map.find(address);
		if (it != _map.end()) {
			return it->_accessSequence;
		} else {
			Node *newNode = new Node(address);
			_map.insert(*newNode); // This operation does actually take the pointer
			return newNode->_accessSequence;
		}
	}
	
};


#endif // FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
