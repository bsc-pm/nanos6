#ifndef INTRUSIVE_LINEAR_REGION_TREAP_HPP
#define INTRUSIVE_LINEAR_REGION_TREAP_HPP

#include <utility>

#include <boost/intrusive/treap_set.hpp>
#include <boost/intrusive/options.hpp>
#include <boost/version.hpp>
#include "DataAccessRange.hpp"


namespace IntrusiveLinearRegionTreapInternals {
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
			return node.getAccessRange().getStartAddressConstRef();
		}
#else
		typedef void *type;
		
		type const &operator()(ContentType const &node)
		{ 
			return node.getAccessRange().getStartAddressConstRef();
		}
#endif
	};
}

//typedef boost::intrusive::treap_set< CacheTrackingObject, boost::intrusive::compare<std::greater<CacheTrackingObject> > > IntrusiveCacheTrackingTreap ;
template <typename ContentType, class Compare/*, class Priority*/>
class IntrusiveLinearRegionTreap : public
	boost::intrusive::treap_set<
		ContentType,
		boost::intrusive::key_of_value<IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>>
	>
{
private:
	typedef boost::intrusive::treap_set<
		ContentType,
		boost::intrusive::key_of_value<IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>>
	> BaseType;
	
public:
	typedef typename BaseType::iterator iterator;
	typedef typename BaseType::const_iterator const_iterator;

	IntrusiveLinearRegionTreap(): BaseType()
	{
	}
	
	const_iterator find(DataAccessRange const &range) const
	{
		return BaseType::find(range.getStartAddress());
	}
	
	iterator find(DataAccessRange const &range)
	{
		return BaseType::find(range.getStartAddress());
	}
	
	void clear()
	{
		for (auto it = BaseType::begin(); it != BaseType::end(); ) {
			it = BaseType::erase(it);
		}
	}
	
	void erase(ContentType &node)
	{
		const_iterator position = BaseType::iterator_to(node);
		BaseType::erase(position);
	}
	void erase(ContentType *node)
	{
		const_iterator position = BaseType::iterator_to(*node);
		BaseType::erase(position);
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
	void processAllWithRearrangement(ProcessorType processor);
	
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
	//! \param[in] missingProcessor a lambda that receives each missing subrange as a DataAccessRange and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename IntersectionProcessorType, typename MissingProcessorType>
	bool processIntersectingAndMissing(DataAccessRange const &range, IntersectionProcessorType intersectingProcessor, MissingProcessorType missingProcessor);
	
	//! \brief Pass all elements that intersect a given range through a lambda with the posibility of restarting
	//! the traversal from the last location if instructed
	//! 
	//! \param[in] range the range to explore
	//! \param[in] processor a lambda that receives an iterator to each element intersecting
	//! the range and that returns a boolean equal to false to have the traversal restart from the current
	//! logical position (since the contents may have changed)
	template <typename ProcessorType>
	void processIntersectingWithRestart(DataAccessRange const &range, ProcessorType processor);
	
	//! \brief Pass any missing subranges through a lambda
	//! 
	//! \param[in] range the range to explore
	//! \param[in] missingProcessor a lambda that receives each missing subrange as a DataAccessRange and that returns a boolean equal to false to stop the traversal
	//! 
	//! \returns false if the traversal was stopped before finishing
	template <typename MissingProcessorType>
	bool processMissing(DataAccessRange const &range, MissingProcessorType missingProcessor);
	
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
	
	//! \brief Fragment an already existing node by the intersection of a given range
	//! 
	//! \param[in] position an iterator to the node to be fragmented
	//! \param[in] range the DataAccessRange that determines the fragmentation point(s)
	//! \param[in] removeIntersection true if the intersection is to be left empty
	//! \param[in] duplicator a lambda that receives a reference to a node and returns a pointer to a new copy
	//! \param[in] postprocessor a lambda that receives a pointer to each node after it has had its range corrected and has been inserted, and a pointer to the original node (that may have already been updated)
	//! 
	//! \returns an iterator to the intersecting fragment or end() if removeIntersection is true
	template <typename DuplicatorType, typename PostProcessorType>
	iterator fragmentByIntersection(iterator position, DataAccessRange const &range, bool removeIntersection, DuplicatorType duplicator, PostProcessorType postprocessor);
	
	//! \brief Fragment any node that intersects by a intersection boundary
	//! 
	//! \param[in] range the DataAccessRange that determines the fragmentation point(s)
	//! \param[in] duplicator a lambda that receives a reference to a node and returns a pointer to a new copy
	//! \param[in] postprocessor a lambda that receives a pointer to each node after it has had its range corrected and has been inserted, and a pointer to the original node (that may have already been updated)
	template <typename DuplicatorType, typename PostProcessorType>
	void fragmentIntersecting(DataAccessRange const &range, DuplicatorType duplicator, PostProcessorType postprocessor);
	
	
	void replace(ContentType &toBeReplaced, ContentType &replacement)
	{
		erase(toBeReplaced);
		BaseType::insert(replacement);
// 		iterator position = BaseType::iterator_to(toBeReplaced);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(replacement)));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(toBeReplaced)));
// 		BaseType::replace_node(position, replacement);
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(replacement)));
// 		BaseType::node_algorithms::init(BaseType::value_traits::to_node_ptr(BaseType::value_traits::to_node_ptr(replacement)));
	}
	void replace(ContentType *toBeReplaced, ContentType *replacement)
	{
		erase(toBeReplaced);
		BaseType::insert(*replacement);
// 		iterator position = BaseType::iterator_to(*toBeReplaced);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(*replacement)));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*toBeReplaced)));
// 		BaseType::replace_node(position, *replacement);
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*replacement)));
// 		BaseType::node_algorithms::init(BaseType::value_traits::to_node_ptr(*toBeReplaced));
	}
	void replace(iterator toBeReplaced, ContentType &replacement)
	{
		erase(*toBeReplaced);
		BaseType::insert(replacement);
// 		assert(BaseType::node_algorithms::inited(BaseType::value_traits::to_node_ptr(replacement)));
// 		assert(BaseType::header_ptr() == BaseType::node_algorithms::get_header(BaseType::value_traits::to_node_ptr(*toBeReplaced)));
// 		BaseType::replace_node(toBeReplaced, replacement);
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

#endif // INTRUSIVE_LINEAR_REGION_TREAP_HPP
