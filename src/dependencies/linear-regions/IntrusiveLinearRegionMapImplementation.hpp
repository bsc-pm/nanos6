/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INTRUSIVE_LINEAR_REGION_MAP_IMPLEMENTATION_HPP
#define INTRUSIVE_LINEAR_REGION_MAP_IMPLEMENTATION_HPP


#include <cassert>
#include <mutex>

#include "IntrusiveLinearRegionMap.hpp"


template <typename ContentType, class Hook> template <typename ProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processAll(ProcessorType processor)
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

template <typename ContentType, class Hook> template <typename ProcessorType>
void IntrusiveLinearRegionMap<ContentType, Hook>::processAllWithRestart(ProcessorType processor)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
			IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!cont) {
			it = BaseType::find(positionIdentifier);
			assert(it != BaseType::end());
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
}


template <typename ContentType, class Hook> template <typename ProcessorType>
void IntrusiveLinearRegionMap<ContentType, Hook>::processAllWithRearangement(ProcessorType processor)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	for (iterator it = BaseType::begin(); it != BaseType::end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		// Keep an identifier for the next position so that the traversal can be restarted from there
		bool nextIsEnd = (it == BaseType::end());
		typename IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier;
		if (!nextIsEnd) {
			positionIdentifier = IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>()(*it);
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


template <typename ContentType, class Hook> template <typename ProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processIntersecting(
	DataAccessRegion const &region,
	ProcessorType processor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (!region.intersect(position->getAccessRegion()).empty()) {
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

template <typename ContentType, class Hook> template <typename ProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processIntersectingWithRecentAdditions(
	DataAccessRegion const &region,
	ProcessorType processor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		iterator position = it;
		
		if (!region.intersect(position->getAccessRegion()).empty()) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				return false;
			}
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		++it;
		
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	}
	
	return true;
}


template <typename ContentType, class Hook> template <typename ProcessorType>
void IntrusiveLinearRegionMap<ContentType, Hook>::processIntersectingWithRestart(
	DataAccessRegion const &region,
	ProcessorType processor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		// Keep an identifier for the current position so that the traversal can be restarted from there
		typename IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>::type positionIdentifier =
		IntrusiveLinearRegionMapInternals::KeyOfNodeArtifact<ContentType>()(*position);
		
		if (!region.intersect(position->getAccessRegion()).empty()) {
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


template <typename ContentType, class Hook> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processIntersectingAndMissing(
	DataAccessRegion const &region,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (BaseType::empty()) {
		return missingProcessor(region); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	iterator initial = it;
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = region.getStartAddress();
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	assert(!BaseType::empty());
	if (it->getAccessRegion().getEndAddress() <= region.getStartAddress()) {
		it = initial;
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		bool cont = true;
		
		// The "processor" may replace the node with something else, so advance before that happens
		iterator position = it;
		it++;
		
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (lastEnd < position->getAccessRegion().getStartAddress()) {
			DataAccessRegion missingRegion(lastEnd, position->getAccessRegion().getStartAddress());
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				return false;
			}
		}
		
		if (position->getAccessRegion().getEndAddress() <= region.getEndAddress()) {
			lastEnd = position->getAccessRegion().getEndAddress();
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		} else {
			assert(position->getAccessRegion().getEndAddress() > region.getEndAddress());
			assert((position->getAccessRegion().getStartAddress() >= lastEnd) || (position->getAccessRegion().getStartAddress() < region.getStartAddress()));
			
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			lastEnd = region.getEndAddress();
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < region.getEndAddress()) {
		DataAccessRegion missingRegion(lastEnd, region.getEndAddress());
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		bool result = missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		return result;
	}
	
	return true;
}


template <typename ContentType, class Hook> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processIntersectingAndMissingWithRecentAdditions(
	DataAccessRegion const &region,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (BaseType::empty()) {
		return missingProcessor(region); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	iterator initial = it;
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = region.getStartAddress();
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	assert(!BaseType::empty());
	if (it->getAccessRegion().getEndAddress() <= region.getStartAddress()) {
		it = initial;
	}
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		bool cont = true;
		
		iterator position = it;
		
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (lastEnd < position->getAccessRegion().getStartAddress()) {
			DataAccessRegion missingRegion(lastEnd, position->getAccessRegion().getStartAddress());
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!cont) {
				return false;
			}
		}
		
		if (position->getAccessRegion().getEndAddress() <= region.getEndAddress()) {
			lastEnd = position->getAccessRegion().getEndAddress();
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		} else {
			assert(position->getAccessRegion().getEndAddress() > region.getEndAddress());
			assert((position->getAccessRegion().getStartAddress() >= lastEnd) || (position->getAccessRegion().getStartAddress() < region.getStartAddress()));
			
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			lastEnd = region.getEndAddress();
		}
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		++it;
		
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < region.getEndAddress()) {
		DataAccessRegion missingRegion(lastEnd, region.getEndAddress());
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		bool result = missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		return result;
	}
	
	return true;
}


template <typename ContentType, class Hook> template <typename MissingProcessorType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::processMissing(
	DataAccessRegion const &region,
	MissingProcessorType missingProcessor
) {
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	return processIntersectingAndMissing(
		region,
		[&](__attribute__((unused)) iterator position) -> bool { return true; },
		missingProcessor
	);
}


template <typename ContentType, class Hook> template <typename PredicateType>
bool IntrusiveLinearRegionMap<ContentType, Hook>::exists(DataAccessRegion const &region, PredicateType condition)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!region.intersect(it->getAccessRegion()).empty()) {
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


template <typename ContentType, class Hook>
bool IntrusiveLinearRegionMap<ContentType, Hook>::contains(DataAccessRegion const &region)
{
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	iterator it = BaseType::lower_bound(region.getStartAddress());
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	if (it != BaseType::begin()) {
		if ((it == BaseType::end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	
	while ((it != BaseType::end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		if (!region.intersect(it->getAccessRegion()).empty()) {
			return true;
		}
		it++;
	}
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	
	return false;
}


template <typename ContentType, class Hook> template <typename DuplicatorType, typename PostProcessorType>
typename IntrusiveLinearRegionMap<ContentType, Hook>::iterator IntrusiveLinearRegionMap<ContentType, Hook>::fragmentByIntersection(
	typename IntrusiveLinearRegionMap<ContentType, Hook>::iterator position,
	DataAccessRegion const &fragmenterRegion,
	bool removeIntersection,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	iterator intersectionPosition = BaseType::end();
	DataAccessRegion originalRegion = position->getAccessRegion();
	bool alreadyShrinked = false;
	ContentType &contents = *position;
	
	assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
	originalRegion.processIntersectingFragments(
		fragmenterRegion,
		/* originalRegion only */
		[&](DataAccessRegion const &region) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			if (!alreadyShrinked) {
				position->setAccessRegion(region);
				alreadyShrinked = true;
				postprocessor(&(*position), &(*position));
				assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			} else {
				ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
				newContents->setAccessRegion(region);
				BaseType::insert(*newContents);
				postprocessor(newContents, &(*position));
				assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			}
		},
		/* intersection */
		[&](DataAccessRegion const &region) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			assert(region == originalRegion.intersect(fragmenterRegion));
			if (!removeIntersection) {
				if (!alreadyShrinked) {
					position->setAccessRegion(region);
					alreadyShrinked = true;
					intersectionPosition = position;
					assert(intersectionPosition->getAccessRegion() == region);
					postprocessor(&(*position), &(*position));
					assert(intersectionPosition->getAccessRegion() == region);
					assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
				} else {
					ContentType *newContents = duplicator(contents); // An error here indicates that the duplicator is missing the "ContentType *" return type
					newContents->setAccessRegion(region);
					intersectionPosition = BaseType::insert(*newContents).first;
					assert(intersectionPosition->getAccessRegion() == region);
					postprocessor(newContents, &(*position));
					assert(intersectionPosition->getAccessRegion() == region);
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
		/* fragmeterRegion only */
		[&](__attribute__((unused)) DataAccessRegion const &region) {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
		}
	);
	
	assert((intersectionPosition == BaseType::end()) || (intersectionPosition->getAccessRegion() == originalRegion.intersect(fragmenterRegion)));
	return intersectionPosition;
}


template <typename ContentType, class Hook> template <typename DuplicatorType, typename PostProcessorType>
void IntrusiveLinearRegionMap<ContentType, Hook>::fragmentIntersecting(
	DataAccessRegion const &region,
	DuplicatorType duplicator,
	PostProcessorType postprocessor
) {
	processIntersecting(
		region,
		[&](iterator position) -> bool {
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			fragmentByIntersection(position, region, false, duplicator, postprocessor);
			assert(BaseType::node_algorithms::verify(BaseType::header_ptr()));
			return true;
		}
	);
}


#endif // INTRUSIVE_LINEAR_REGION_MAP_IMPLEMENTATION_HPP
