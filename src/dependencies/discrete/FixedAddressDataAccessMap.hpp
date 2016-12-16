#ifndef FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
#define FIXED_ADDRESS_DATA_ACCESS_MAP_HPP


#include <utility>

#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/version.hpp>

#include "DataAccessRange.hpp"
#include "RootDataAccessSequence.hpp"
#include "RootDataAccessSequenceLinkingArtifacts.hpp"
#include "RootDataAccessSequenceLinkingArtifactsImplementation.hpp"


namespace FixedAddressDataAccessMapInternals {
	template <typename ContentType>
	struct KeyOfNodeArtifact
	{
#if BOOST_VERSION >= 106200
		typedef DataAccessRange type;
		
		type operator()(ContentType const &node)
		{ 
			return node.getAccessRange();
		}
#else
		typedef DataAccessRange type;
		
		type const &operator()(ContentType const &node)
		{ 
			return node.getAccessRange();
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
	RootDataAccessSequence &operator[](DataAccessRange accessRange)
	{
		auto it = BaseType::find(accessRange);
		if (it != BaseType::end()) {
			return *it;
		} else {
			RootDataAccessSequence *newNode = new RootDataAccessSequence(accessRange);
			BaseType::insert(*newNode);
			return *newNode;
		}
	}
};


#endif // FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
