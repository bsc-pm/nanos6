/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
#define FIXED_ADDRESS_DATA_ACCESS_MAP_HPP


#include <utility>

#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/version.hpp>

#include "DataAccessRegion.hpp"
#include "RootDataAccessSequence.hpp"
#include "RootDataAccessSequenceLinkingArtifacts.hpp"
#include "RootDataAccessSequenceLinkingArtifactsImplementation.hpp"


namespace FixedAddressDataAccessMapInternals {
	template <typename ContentType>
	struct KeyOfNodeArtifact
	{
#if BOOST_VERSION >= 106200
		typedef DataAccessRegion type;
		
		type operator()(ContentType const &node)
		{ 
			return node.getAccessRegion();
		}
#else
		typedef DataAccessRegion type;
		
		type const &operator()(ContentType const &node)
		{ 
			return node.getAccessRegion();
		}
#endif
	};
}


class FixedAddressDataAccessMap
	: public boost::intrusive::avl_set<
		RootDataAccessSequence,
		boost::intrusive::key_of_value<FixedAddressDataAccessMapInternals::KeyOfNodeArtifact<RootDataAccessSequence>>,
		boost::intrusive::function_hook<RootDataAccessSequenceLinkingArtifacts>
	>
{
private:
	typedef boost::intrusive::avl_set<
		RootDataAccessSequence,
		boost::intrusive::key_of_value<FixedAddressDataAccessMapInternals::KeyOfNodeArtifact<RootDataAccessSequence>>,
		boost::intrusive::function_hook<RootDataAccessSequenceLinkingArtifacts>
	> BaseType;
	
public:
	RootDataAccessSequence &operator[](DataAccessRegion accessRegion)
	{
		auto it = BaseType::find(accessRegion);
		if (it != BaseType::end()) {
			return *it;
		} else {
			RootDataAccessSequence *newNode = new RootDataAccessSequence(accessRegion);
			BaseType::insert(*newNode);
			return *newNode;
		}
	}
};


#endif // FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
