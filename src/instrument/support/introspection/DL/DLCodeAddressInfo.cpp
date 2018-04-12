/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include "DLCodeAddressInfo.hpp"

#include <cassert>
#include <sstream>


std::map<void *, CodeAddressInfoBase::Entry> DLCodeAddressInfo::_address2Entry;


void DLCodeAddressInfo::init()
{
}


void DLCodeAddressInfo::shutdown()
{
	_address2Entry.clear();
}


CodeAddressInfoBase::Entry const &DLCodeAddressInfo::resolveAddress(void *address)
{
	// Check in the cache
	auto it = _address2Entry.find(address);
	if (it != _address2Entry.end()) {
		return it->second;
	}
	
	Dl_info dlInfo;
	
	int rc = dladdr(address, &dlInfo);
	if (rc == 0) {
		// Error
		return _nullEntry;
	}
	
	if (dlInfo.dli_sname == nullptr) {
		// Not found
		return _nullEntry;
	}
	
	// Create the entry
	Entry &entry = _address2Entry[address];
	
	// Add the current function and source location
	{
		std::string mangledFunction = dlInfo.dli_sname;
		std::string function = DLCodeAddressInfo::demangleSymbol(mangledFunction);
		std::string sourceLine;
		
		if (address != dlInfo.dli_saddr) {
			std::ostringstream oss;
			oss << function << "+0x" << + std::hex << (void *)((size_t) address - (size_t) dlInfo.dli_saddr);
			sourceLine = oss.str();
			
			mangledFunction = mangledFunction + std::string(" [inside]");
			function = function + std::string(" [inside]");
		} else {
			sourceLine = function;
		}
		
		InlineFrame currentFrame = functionAndSourceToFrame(mangledFunction, function, sourceLine);
		entry._inlinedFrames.push_back(currentFrame);
	}
	
	return entry;
}
