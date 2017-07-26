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
	DataAccessRange const &range,
	ProcessorType processor
) {
	iterator it = _map.lower_bound(range.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRange().getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != _map.end()) && (it->getAccessRange().getStartAddress() < range.getEndAddress())) {
		// The "processor" may replace the node by something else, so advance before that happens
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


template <typename ContentType> template <typename IntersectingProcessorType, typename MissingProcessorType>
bool LinearRegionMap<ContentType>::processIntersectingAndMissing(
	DataAccessRange const &range,
	IntersectingProcessorType intersectingProcessor,
	MissingProcessorType missingProcessor
) {
	if (_map.empty()) {
		return missingProcessor(range); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	iterator it = _map.lower_bound(range.getStartAddress());
	iterator initial = it;
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRange().getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	void *lastEnd = range.getStartAddress();
	assert(!_map.empty());
	if (it->getAccessRange().getEndAddress() <= range.getStartAddress()) {
		it = initial;
	}
	
	while ((it != _map.end()) && (it->getAccessRange().getStartAddress() < range.getEndAddress())) {
		bool cont = true;
		
		// The "processor" may replace the node by something else, so advance before that happens
		iterator position = it;
		it++;
		
		if (lastEnd < position->getAccessRange().getStartAddress()) {
			DataAccessRange missingRange(lastEnd, position->getAccessRange().getStartAddress());
			cont = missingProcessor(missingRange); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			if (!cont) {
				return false;
			}
		}
		
		if (position->getAccessRange().getEndAddress() <= range.getEndAddress()) {
			lastEnd = position->getAccessRange().getEndAddress();
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
		} else {
			assert(position->getAccessRange().getEndAddress() > range.getEndAddress());
			assert((position->getAccessRange().getStartAddress() >= lastEnd) || (position->getAccessRange().getStartAddress() < range.getStartAddress()));
			
			cont = intersectingProcessor(position); // NOTE: an error here indicates that the lambda is missing the "bool" return type
			lastEnd = range.getEndAddress();
		}
		
		if (!cont) {
			return false;
		}
	}
	
	if (lastEnd < range.getEndAddress()) {
		DataAccessRange missingRange(lastEnd, range.getEndAddress());
		return missingProcessor(missingRange); // NOTE: an error here indicates that the lambda is missing the "bool" return type
	}
	
	return true;
}


template <typename ContentType> template <typename PredicateType>
bool LinearRegionMap<ContentType>::exists(DataAccessRange const &range, PredicateType condition)
{
	iterator it = _map.lower_bound(range.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRange().getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	
	while ((it != _map.end()) && (it->getAccessRange().getStartAddress() < range.getEndAddress())) {
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


template <typename ContentType>
bool LinearRegionMap<ContentType>::contains(DataAccessRange const &range)
{
	iterator it = _map.lower_bound(range.getStartAddress());
	
	if (it != _map.begin()) {
		if ((it == _map.end()) || (it->getAccessRange().getStartAddress() > range.getStartAddress())) {
			it--;
		}
	}
	
	while ((it != _map.end()) && (it->getAccessRange().getStartAddress() < range.getEndAddress())) {
		if (!range.intersect(it->getAccessRange()).empty()) {
			return true;
		}
		it++;
	}
	
	return false;
}


template <typename ContentType>
typename LinearRegionMap<ContentType>::iterator LinearRegionMap<ContentType>::fragmentByIntersection(
	typename LinearRegionMap<ContentType>::iterator position,
	DataAccessRange const &fragmenterRange,
	bool removeIntersection
) {
	iterator intersectionPosition = end();
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
			} else {
				ContentType newContents(contents);
				newContents.getAccessRange() = range;
				insert(newContents);
			}
		},
		/* intersection */
		[&](DataAccessRange const &range) {
			if (!removeIntersection) {
				if (!alreadyShrinked) {
					position->getAccessRange() = range;
					alreadyShrinked = true;
					intersectionPosition = position;
				} else {
					ContentType newContents(contents);
					newContents.getAccessRange() = range;
					intersectionPosition = insert(newContents);
				}
			} else {
				if (!alreadyShrinked) {
					erase(position);
					alreadyShrinked = true;
				}
			}
		},
		/* fragmeterRange only */
		[&](__attribute__((unused)) DataAccessRange const &range) {
		}
	);
	
	return intersectionPosition;
}


#endif // LINEAR_REGION_MAP_IMPLEMENTATION_HPP
