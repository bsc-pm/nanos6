#ifndef INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP
#define INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP


#include <cassert>
#include <mutex>

#include "IntrusiveLinearRegionTreap.hpp"


template <typename ContentType> template <typename ProcessorType>
bool IntrusiveLinearRegionTreap<ContentType>::processAll(ProcessorType processor)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!cont) {
			return false;
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	return true;
}


template <typename ContentType> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType>::processAllWithRestart(ProcessorType processor)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
			IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!cont) {
			it = BaseType::find(positionIdentifier);
			assert(it != BaseType::end());
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
}


template <typename ContentType> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType>::processAllWithRearrangement(ProcessorType processor)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		// Keep an identifier for the next position so that the traversal can be restarted from there
		bool nextIsEnd = (it == BaseType::end());
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier;
		if (!nextIsEnd) {
			positionIdentifier = IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*it);
		}
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!cont) {
			if (nextIsEnd) {
				return;
			}
			it = BaseType::lower_bound(positionIdentifier);
			// The next could end up being end() since the processor can have removed the remaining nodes
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
}


template <typename ContentType> template <typename ProcessorType>
bool IntrusiveLinearRegionTreap<ContentType>::processIntersecting(
	DataAccessRange const &range,
	ProcessorType processor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(range.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->_range.getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->_range.getStartAddress() < range.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (!range.intersect(position->_range).empty()) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				return false;
			}
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	}
	
	return true;
}


template <typename ContentType> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType>::processIntersectingWithRestart(
	DataAccessRange const &range,
	ProcessorType processor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(range.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->_range.getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->_range.getStartAddress() < range.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
		IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		if (!range.intersect(position->_range).empty()) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				it = BaseType::find(positionIdentifier);
				assert(it != BaseType::end());
			}
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	}
}


template <typename ContentType> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool IntrusiveLinearRegionTreap<ContentType>::processIntersectingAndMissing(
	DataAccessRange const &range,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (BaseType::empty()) {
		return missingProcessor(range); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(range.getStartAddress());
	iterator initial = it;
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->_range.getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = range.getStartAddress();
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	assert(!BaseType::empty());
	if (it->_range.getEndAddress() <= range.getStartAddress()) {
		it = initial;
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->_range.getStartAddress() < range.getEndAddress())) {
		bool cont = true;
		
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (lastEnd < position->_range.getStartAddress()) {
			DataAccessRange missingRange(lastEnd, position->_range.getStartAddress());
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = missingProcessor(missingRange); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				return false;
			}
		}
		
		if (position->_range.getEndAddress() <= range.getEndAddress()) {
			lastEnd = position->_range.getEndAddress();
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		} else {
			assert(position->_range.getEndAddress() > range.getEndAddress());
			assert((position->_range.getStartAddress() >= lastEnd) || (position->_range.getStartAddress() < range.getStartAddress()));
			
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			lastEnd = range.getEndAddress();
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < range.getEndAddress()) {
		DataAccessRange missingRange(lastEnd, range.getEndAddress());
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		bool result = missingProcessor(missingRange); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		return result;
	}
	
	return true;
}


template <typename ContentType> template <typename MissingProcessorType>
bool IntrusiveLinearRegionTreap<ContentType>::processMissing(
	DataAccessRange const &range,
	MissingProcessorType missingProcessor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	return processIntersectingAndMissing(
		range,
		[&](__attribute__((unused)) iterator position) -> bool { return true; },
		missingProcessor
	);
}


template <typename ContentType> template <typename PredicateType>
bool IntrusiveLinearRegionTreap<ContentType>::exists(DataAccessRange const &range, PredicateType condition)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(range.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->_range.getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	
	while ((it != BaseType::end()) && (it->_range.getStartAddress() < range.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!range.intersect(it->_range).empty()) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			bool found = condition(it); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (found) {
				return true;
			}
		}
		it++;
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	return false;
}


template <typename ContentType>
bool IntrusiveLinearRegionTreap<ContentType>::contains(DataAccessRange const &range)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(range.getStartAddress());
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->_range.getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	
	while ((it != BaseType::end()) && (it->_range.getStartAddress() < range.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!range.intersect(it->_range).empty()) {
			return true;
		}
		it++;
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	return false;
}


template <typename ContentType> template <typename DuplicatorType, typename PostProcessorType>
typename IntrusiveLinearRegionTreap<ContentType>::iterator IntrusiveLinearRegionTreap<ContentType>::fragmentByIntersection(
	typename IntrusiveLinearRegionTreap<ContentType>::iterator position,
	DataAccessRange const &fragmenterRange,
	bool removeIntersection,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	iterator intersectionPosition = BaseType::end();
	DataAccessRange originalRange = position->_range;
	bool alreadyShrinked = false;
	ContentType &contents = *position;
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	originalRange.processIntersectingFragments(
		fragmenterRange,
		/* originalRange only */
		[&](DataAccessRange const &range) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!alreadyShrinked) {
				position->_range = range;
				alreadyShrinked = true;
				postprocessor(&(*position), &(*position));
				assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			} else {
				ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
				newContents->getAccessRange() = range;
				BaseType::insert(*newContents);
				postprocessor(newContents, &(*position));
				assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			}
		},
		/* intersection */
		[&](DataAccessRange const &range) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			assert(range == originalRange.intersect(fragmenterRange));
			if (!removeIntersection) {
				if (!alreadyShrinked) {
					position->_range = range;
					alreadyShrinked = true;
					intersectionPosition = position;
					assert(intersectionPosition->_range == range);
					postprocessor(&(*position), &(*position));
					assert(intersectionPosition->_range == range);
					assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
				} else {
					ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
					newContents->getAccessRange() = range;
					intersectionPosition = BaseType::insert(*newContents).first;
					assert(intersectionPosition->_range == range);
					postprocessor(newContents, &(*position));
					assert(intersectionPosition->_range == range);
					assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
				}
			} else {
				if (!alreadyShrinked) {
					assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
					BaseType::erase(position);
					assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
					alreadyShrinked = true;
				}
			}
		},
		/* fragmeterRange only */
		[&](__attribute__((unused)) DataAccessRange const &range) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		}
	);
	
	assert((intersectionPosition == BaseType::end()) || (intersectionPosition->_range == originalRange.intersect(fragmenterRange)));
	return intersectionPosition;
}


template <typename ContentType> template <typename DuplicatorType, typename PostProcessorType>
void IntrusiveLinearRegionTreap<ContentType>::fragmentIntersecting(
	DataAccessRange const &range,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	processIntersecting(
		range,
		[&](iterator position) -> bool {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			fragmentByIntersection(position, range, false, duplicator, postprocessor);
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			return true;
		}
	);
}


#endif // INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP
