/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_INSTROSPECTION_ADDR2LINE_CODE_ADDRESS_INFO_HPP
#define INSTRUMENT_SUPPORT_INSTROSPECTION_ADDR2LINE_CODE_ADDRESS_INFO_HPP


#include <string>

#include "../CodeAddressInfoBase.hpp"


class Addr2LineCodeAddressInfo : public CodeAddressInfoBase {
private:
	// Map between the address space and the executable objects
	struct MemoryMapSegment {
		std::string _filename;
		size_t _offset;
		size_t _length;
		
		MemoryMapSegment()
			: _filename(), _offset(0), _length(0)
		{
		}
	};
	
	static std::map<void *, MemoryMapSegment> _executableMemoryMap;
	
public:
	static void init();
	static void shutdown();
	static Entry const &resolveAddress(void *address, bool callSiteFromReturnAddress = false);
	
};


#endif // INSTRUMENT_SUPPORT_INSTROSPECTION_ADDR2LINE_CODE_ADDRESS_INFO_HPP
