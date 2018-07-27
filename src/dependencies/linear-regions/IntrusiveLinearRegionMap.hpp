/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INTRUSIVE_LINEAR_REGION_MAP_HPP
#define INTRUSIVE_LINEAR_REGION_MAP_HPP

#include <utility>

#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/version.hpp>
#include "DataAccessRegion.hpp"


namespace IntrusiveLinearRegionMapInternals {
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
		
		type operator()(ContentType const &node)
		{ 
			return node.getAccessRegion().getStartAddressConstRef();
		}
#else
		typedef void *type;
		
		type const &operator()(ContentType const &node)
		{ 
			return node.getAccessRegion().getStartAddressConstRef();
		}
#endif
	};
}


template <typename ContentType, class Hook>
class IntrusiveLinearRegionMap : public
	boost::intrusive::avl_set<
		ContentType,
		boost::intrusive::key_of_value<IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>>,
		Hook
	>
{
private:
	typedef boost::intrusive::avl_set<
		ContentType,
		boost::intrusive::key_of_value<IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>>,
		Hook
	> BaseType;
	
public:
	typedef typename BaseType::iterator iterator;
	typedef typename BaseType::const_iterator const_iterator;
	
	
	IntrusiveLinearRegionMap(): BaseType()
	{
	}
	
	const_iterator find(DataAccessRegion const &region) const
	{
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		return BaseType::find(region.getStartAddress());
	}
	
	iterator find(DataAccessRegion const &region)
	{
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		return BaseType::find(region.getStartAddress());
	}
	
	void clear()
	{
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		for (auto it = BaseType::begin(); it != BaseType::end(); ) {
			it = BaseType::erase(it);
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		}
	}
	
	void erase(ContentType &victim)
	{
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		const_iterator position = BaseType::iterator_to(victim);
		BaseType::erase(position);
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	}
	void erase(ContentType *victim)
	{
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		const_iterator position = BaseType::iterator_to(*victim);
		BaseType::erase(position);
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	}
	
	
	
	//! \brief Pass all elements through a lambda
	//! 
	//! \param[in] processor a lambda that receives an iterator to each element that returns a boolean that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processAll(ProcessorType processor);
	
	//! \brief Pass all elements through a lambda and restart from the last location if instructed
	//! 
	//! \param[in] processor a lambda that receives an iterator to each element that returns a boolean that is false to have the traversal restart from the current logical position (since the contents may have changed)
	template <typename ProcessorType>
	void processAllWithRestart(ProcessorType processor);
	
	//! \brief Pass all elements through a lambda but accept changes to the whole contents if instructed
	//! 
	//! \param[in] processor a lambda that receives an iterator to each element that returns a boolean that is false to have the traversal restart from the next logical position in the event of invasive content changes
	template <typename ProcessorType>
	void processAllWithRearangement(ProcessorType processor);
	
	//! \brief Pass all elements that intersect a given region through a lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting the region and that returns a boolean, that is false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processIntersecting(DataAccessRegion const &region, ProcessorType processor);
	
	//! \brief Pass all elements that intersect a given region through a lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting the region and that returns a boolean, that is false to stop the traversal. Unless the processor returns false, it should not invalidate the iterator passed as a parameter
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	bool processIntersectingWithRecentAdditions(DataAccessRegion const &region, ProcessorType processor);
	
	//! \brief Pass all elements that intersect a given region through a lambda and any missing subregions through another lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] intersectingProcessor a lambda that receives an iterator to each element intersecting the region and that returns a boolean equal to false to stop the traversal
	//! \param[in] missingProcessor a lambda that receives each missing subregion as a DataAccessRegion and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissing(DataAccessRegion const &region, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor);
	
	//! \brief Pass all elements that intersect a given region through a lambda and any missing subregions through another lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] intersectingProcessor a lambda that receives an iterator to each element intersecting the region and that returns a boolean equal to false to stop the traversal. Unless the processor returns false, it should not invalidate the iterator passed as a parameter
	//! \param[in] missingProcessor a lambda that receives each missing subregion as a DataAccessRegion and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissingWithRecentAdditions(DataAccessRegion const &region, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor);
	
	//! \brief Pass all elements that intersect a given region through a lambda with the posibility of restarting
	//! the traversal from the last location if instructed
	//! 
	//! \param[in] region the region to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting
	//! the region and that returns a boolean equal to false to have the traversal restart from the current
	//! logical position (since the contents may have changed)
	template <typename ProcessorType>
	void processIntersectingWithRestart(DataAccessRegion const &region, ProcessorType processor);
	
	//! \brief Pass any missing subregions through a lambda
	//! 
	//! \param[in] region the region to explore
	//! \param[in] missingProcessor a lambda that receives each missing subregion as a DataAccessRegion and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename MissingProcessorType>
	bool processMissing(DataAccessRegion const &region, MissingProcessorType missingProcessor);
	
	//! \brief Traverse a region of elements to check if there is an element that matches a given condition
	//! 
	//! \param[in] region the region to explore
	//! \param[in] condition a lambda that receives an iterator to each element intersecting the region and that returns the result of evaluating the condition
	//! 
	//! \returns true if the condition evaluated to true for any element
	template <typename PredicateType>
	bool exists(DataAccessRegion const &region, PredicateType condition);
	
	//! \brief Check if there is any element in a given region
	//! 
	//! \param[in] region the region to explore
	//! 
	//! \returns true if there was at least one element at least partially in the region
	bool contains(DataAccessRegion const &region);
	
	//! \brief Fragment an already existing node by the intersection of a given region
	//! 
	//! \param[in] position an iterator to the node to be fragmented
	//! \param[in] region the DataAccessRegion that determines the fragmentation point(s)
	//! \param[in] removeIntersection true if the intersection is to be left empty
	//! \param[in] duplicator a lambda that receives a reference to a node and returns a pointer to a new copy
	//! \param[in] postprocessor a lambda that receives a pointer to each node after it has had its region corrected and has been inserted, and a pointer to the original node (that may have already been updated)
	//! 
	//! \returns an iterator to the intersecting fragment or end() if removeIntersection is true
	template <typename DuplicatorType, typename PostProcessorType>
	iterator fragmentByIntersection(iterator position, DataAccessRegion const &region, bool removeIntersection, DuplicatorType duplicator, PostProcessorType postprocessor);
	
	//! \brief Fragment any node that intersects by a intersection boundary
	//! 
	//! \param[in] region the DataAccessRegion that determines the fragmentation point(s)
	//! \param[in] duplicator a lambda that receives a reference to a node and returns a pointer to a new copy
	//! \param[in] postprocessor a lambda that receives a pointer to each node after it has had its region corrected and has been inserted, and a pointer to the original node (that may have already been updated)
	template <typename DuplicatorType, typename PostProcessorType>
	void fragmentIntersecting(DataAccessRegion const &region, DuplicatorType duplicator, PostProcessorType postprocessor);
	
	
	void replace(ContentType &toBeReplaced, ContentType &replacement)
	{
		erase(toBeReplaced);
		BaseType::insert(replacement);
// 		iterator position = BaseType::iterator_to(toBeReplaced);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(replacement)));
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(toBeReplaced)));
// 		BaseType::replace_node(position, replacement);
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(replacement)));
// 		BaseType::node_algorithms::init(BaseType::value_traits::to_node_ptr(BaseType::value_traits::to_node_ptr(replacement)));
	}
	void replace(ContentType *toBeReplaced, ContentType *replacement)
	{
		erase(toBeReplaced);
		BaseType::insert(*replacement);
// 		iterator position = BaseType::iterator_to(*toBeReplaced);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(*replacement)));
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*toBeReplaced)));
// 		BaseType::replace_node(position, *replacement);
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*replacement)));
// 		BaseType::node_algorithms::init(BaseType::value_traits::to_node_ptr(*toBeReplaced));
	}
	void replace(iterator toBeReplaced, ContentType &replacement)
	{
		erase(*toBeReplaced);
		BaseType::insert(replacement);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(replacement)));
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*toBeReplaced)));
// 		BaseType::replace_node(toBeReplaced, replacement);
// 		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(replacement)));
// 		BaseType::node_algorithms::init(BaseType::value_traits::to_node_ptr(*toBeReplaced));
	}
	
	//! \brief Delete all the elements
	//! 
	//! \param[in] processor a lambda that receives a pointer to each element to dispose it
	template <typename ProcessorType>
	void deleteAll(ProcessorType processor)
	{
		BaseType::clear_and_dispose(processor);
	}
	
};



#endif // INTRUSIVE_LINEAR_REGION_MAP_HPP
