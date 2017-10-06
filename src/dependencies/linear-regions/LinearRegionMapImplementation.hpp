/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef LINEAR_REGION_MAP_IMPLEMENTATION_HPP
#define LINEAR_REGION_MAP_IMPLEMENTATION_HPP


#include <cassert>
#include <mutex>

#include "LinearRegionMap.hpp"


template <typename ContentType> template <typename ProcessorType>
bool LinearRegionMap<ContentType>::processAll(ProcessorType processor)
{
	for (iterator it = _map.begin(); it != _map.end(); ) {
		iterator position = it;
		it++; // Advance before processing to allow the processor to fragment the node without passing a second time over some new fragments
		
		bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		if (!cont) {
			return false;
		}
	}
	
	return true;
}


template <typename ContentType> template <typename ProcessorType>
bool LinearRegionMap<ContentType>::processIntersecting(
	DataAccessRegion const &region,
	ProcessorType processor
) {
	iterator it = _map.lower_bound(region.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != _map.end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		// The "processor" may replace the node by something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (!region.intersect(position->getAccessRegion()).empty()) {
			bool cont = processor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
	}
	
	return true;
}


template <typename ContentType> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool LinearRegionMap<ContentType>::processIntersectingAndMissing(
	DataAccessRegion const &region,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	if (_map.empty()) {
		return missingProcessor(region); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	iterator it = _map.lower_bound(region.getStartAddress());
	iterator initial = it;
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = region.getStartAddress();
	assert(!_map.empty());
	if (it->getAccessRegion().getEndAddress() <= region.getStartAddress()) {
		it = initial;
	}
	
	while ((it != _map.end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		bool cont = true;
		
		// The "processor" may replace the node by something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (lastEnd < position->getAccessRegion().getStartAddress()) {
			DataAccessRegion missingRegion(lastEnd, position->getAccessRegion().getStartAddress());
			cont = missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
		
		if (position->getAccessRegion().getEndAddress() <= region.getEndAddress()) {
			lastEnd = position->getAccessRegion().getEndAddress();
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		} else {
			assert(position->getAccessRegion().getEndAddress() > region.getEndAddress());
			assert((position->getAccessRegion().getStartAddress() >= lastEnd) || (position->getAccessRegion().getStartAddress() < region.getStartAddress()));
			
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			lastEnd = region.getEndAddress();
		}
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < region.getEndAddress()) {
		DataAccessRegion missingRegion(lastEnd, region.getEndAddress());
		return missingProcessor(missingRegion); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	return true;
}


template <typename ContentType> template <typename PredicateType>
bool LinearRegionMap<ContentType>::exists(DataAccessRegion const &region, PredicateType condition)
{
	iterator it = _map.lower_bound(region.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	
	while ((it != _map.end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		if (!region.intersect(it->getAccessRegion()).empty()) {
			bool found = condition(it); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (found) {
				return true;
			}
		}
		it++;
	}
	
	return false;
}


template <typename ContentType>
bool LinearRegionMap<ContentType>::contains(DataAccessRegion const &region)
{
	iterator it = _map.lower_bound(region.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRegion().getStartAddress() > region.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != _map.end()) && (it->getAccessRegion().getStartAddress() < region.getEndAddress())) {
		if (!region.intersect(it->getAccessRegion()).empty()) {
			return true;
		}
		it++;
	}
	
	return false;
}


template <typename ContentType>
typename LinearRegionMap<ContentType>::iterator LinearRegionMap<ContentType>::fragmentByIntersection(
	typename LinearRegionMap<ContentType>::iterator position,
	DataAccessRegion const &fragmenterRegion,
	bool removeIntersection
) {
	iterator intersectionPosition = end();
	DataAccessRegion originalRegion = position->getAccessRegion();
	bool alreadyShrinked = false;
	ContentType &contents = *position;
	
	originalRegion.processIntersectingFragments(
		fragmenterRegion,
		/* originalRegion only */
		[&](DataAccessRegion const &region) {
			if (!alreadyShrinked) {
				position->getAccessRegion() = region;
				alreadyShrinked = true;
			} else {
				ContentType newContents(contents);
				newContents.getAccessRegion() = region;
				insert(newContents);
			}
		},
		/* intersection */
		[&](DataAccessRegion const &region) {
			if (!removeIntersection) {
				if (!alreadyShrinked) {
					position->getAccessRegion() = region;
					alreadyShrinked = true;
					intersectionPosition = position;
				} else {
					ContentType newContents(contents);
					newContents.getAccessRegion() = region;
					intersectionPosition = insert(newContents);
				}
			} else {
				if (!alreadyShrinked) {
					erase(position);
					alreadyShrinked = true;
				}
			}
		},
		/* fragmeterRegion only */
		[&](__attribute__((unused)) DataAccessRegion const &region) {
		}
	);
	
	return intersectionPosition;
}


#endif // LINEAR_REGION_MAP_IMPLEMENTATION_HPP
