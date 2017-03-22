#ifndef INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP
#define INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP


#include <cassert>
#include <mutex>

#include "IntrusiveLinearRegionTreap.hpp"


template <typename ContentType, class Hook, class Compare, class Priority> template <typename ProcessorType>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processAll(ProcessorType processor)
{
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		if (!cont) {
			return false;
		}
	}
	
	return true;
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processAllWithRestart(ProcessorType processor)
{
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
			IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		if (!cont) {
			it = BaseType::find(positionIdentifier);
			assert(it != BaseType::end());
		}
	}
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processAllWithRearrangement(ProcessorType processor)
{
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		
		// Keep an identifier for the next position so that the traversal can be restarted from there
		bool nextIsEnd = (it == BaseType::end());
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier;
		if (!nextIsEnd) {
			positionIdentifier = IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*it);
		}
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		if (!cont) {
			if (nextIsEnd) {
				return;
			}
			it = BaseType::lower_bound(positionIdentifier);
			// The next could end up being end() since the processor can have removed the remaining nodes
		}
	}
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename ProcessorType>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processIntersecting(
	DataAccessRange const &range,
	ProcessorType processor
) {
	iterator it = BaseType::lower_bound(IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(range));
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != BaseType::end()) && (it->getStartAddress() < range.getEndAddress())) {
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (!range.intersect(position->getAccessRange()).empty()) {
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
	}
	
	return true;
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename ProcessorType>
void IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processIntersectingWithRestart(
	DataAccessRange const &range,
	ProcessorType processor
) {
	iterator it = BaseType::lower_bound(IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(range));
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != BaseType::end()) && (it->getStartAddress() < range.getEndAddress())) {
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
		IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		if (!range.intersect(position->getAccessRange()).empty()) {
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				it = BaseType::find(positionIdentifier);
				assert(it != BaseType::end());
			}
		}
	}
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processIntersectingAndMissing(
	DataAccessRange const &range,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	if (BaseType::empty()) {
		return missingProcessor(range, BaseType::end()); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}

	iterator it = BaseType::lower_bound(IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(range));
	iterator initial = it;
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = range.getStartAddress();
	assert(!BaseType::empty());
	if (it->getEndAddress() <= range.getStartAddress()) {
		it = initial;
	}
	
	while ((it != BaseType::end()) && (it->getStartAddress() < range.getEndAddress())) {
		bool cont = true;
		
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;

		if (lastEnd < position->getStartAddress()) {
			DataAccessRange missingRange(lastEnd, position->getStartAddress());
			cont = missingProcessor(missingRange, position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			//cont = missingProcessor(missingRange, it); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			//cont = missingProcessor(missingRange); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
		
		if (position->getEndAddress() <= range.getEndAddress()) {
			lastEnd = position->getEndAddress();
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		} else {
			assert(position->getEndAddress() > range.getEndAddress());
			assert((position->getStartAddress() >= lastEnd) || (position->getStartAddress() < range.getStartAddress()));
			
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			lastEnd = range.getEndAddress();
		}
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < range.getEndAddress()) {
		DataAccessRange missingRange(lastEnd, range.getEndAddress());
		bool result = missingProcessor(missingRange, it); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		return result;
	}
	
	return true;
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename MissingProcessorType>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::processMissing(
	DataAccessRange const &range,
	MissingProcessorType missingProcessor
) {
	return processIntersectingAndMissing(
		range,
		[&](__attribute__((unused)) iterator position) -> bool { return true; },
		missingProcessor
	);
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename PredicateType>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::exists(DataAccessRange const &range, PredicateType condition)
{
	iterator it = BaseType::lower_bound(IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(range));
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	
	while ((it != BaseType::end()) && (it->getStartAddress() < range.getEndAddress())) {
		if (!range.intersect(it->getAccessRange()).empty()) {
			bool found = condition(it); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (found) {
				return true;
			}
		}
		it++;
	}
	
	return false;
}


template <typename ContentType, class Hook, class Compare, class Priority>
bool IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::contains(DataAccessRange const &range)
{
	iterator it = BaseType::lower_bound(IntrusiveLinearRegionTreapInternals::KeyOfNodeArtifact<ContentType>()(range));
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	
	while ((it != BaseType::end()) && (it->getStartAddress() < range.getEndAddress())) {
		if (!range.intersect(it->getAccessRange()).empty()) {
			return true;
		}
		it++;
	}
	
	return false;
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename DuplicatorType, typename PostProcessorType>
typename IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::iterator IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::fragmentByIntersection(
	typename IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::iterator position,
	DataAccessRange const &fragmenterRange,
	bool removeIntersection,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	iterator intersectionPosition = BaseType::end();
	DataAccessRange originalRange = position->getAccessRange();
	bool alreadyShrinked = false;
	ContentType &contents = *position;
	
	originalRange.processIntersectingFragments(
		fragmenterRange,
		/* originalRange only */
		[&](DataAccessRange const &range) {
			if (!alreadyShrinked) {
				position->getAccessRange() = range;
				alreadyShrinked = true;
				postprocessor(&(*position), &(*position));
			} else {
				ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
				newContents->getAccessRange() = range;
				BaseType::insert(*newContents);
				postprocessor(newContents, &(*position));
			}
		},
		/* intersection */
		[&](DataAccessRange const &range) {
			assert(range == originalRange.intersect(fragmenterRange));
			if (!removeIntersection) {
				if (!alreadyShrinked) {
					position->getAccessRange() = range;
					alreadyShrinked = true;
					intersectionPosition = position;
					assert(intersectionPosition->getAccessRange() == range);
					postprocessor(&(*position), &(*position));
					assert(intersectionPosition->getAccessRange() == range);
				} else {
					ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
					newContents->getAccessRange() = range;
					intersectionPosition = BaseType::insert(*newContents).first;
					assert(intersectionPosition->getAccessRange() == range);
					postprocessor(newContents, &(*position));
					assert(intersectionPosition->getAccessRange() == range);
				}
			} else {
				if (!alreadyShrinked) {
					BaseType::erase(position);
					alreadyShrinked = true;
				}
			}
		},
		/* fragmeterRange only */
		[&](__attribute__((unused)) DataAccessRange const &range) {
		}
	);
	
	assert((intersectionPosition == BaseType::end()) || (intersectionPosition->getAccessRange() == originalRange.intersect(fragmenterRange)));
	return intersectionPosition;
}


template <typename ContentType, class Hook, class Compare, class Priority> template <typename DuplicatorType, typename PostProcessorType>
void IntrusiveLinearRegionTreap<ContentType, Hook, Compare, Priority>::fragmentIntersecting(
	DataAccessRange const &range,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	processIntersecting(
		range,
		[&](iterator position) -> bool {
			fragmentByIntersection(position, range, false, duplicator, postprocessor);
			return true;
		}
	);
}


#endif // INTRUSIVE_LINEAR_REGION_TREAP_IMPLEMENTATION_HPP
