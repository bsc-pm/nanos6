/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_INSTROSPECTION_DL_CODE_ADDRESS_INFO_HPP
#define INSTRUMENT_SUPPORT_INSTROSPECTION_DL_CODE_ADDRESS_INFO_HPP


#include <map>
#include <string>

#include "../CodeAddressInfoBase.hpp"


class DLCodeAddressInfo : public CodeAddressInfoBase {
private:
	static std::map<void *, Entry> _address2Entry;
	
public:
	static void init();
	static void shutdown();
	static Entry const &resolveAddress(void *address, bool callSiteFromReturnAddress = false);
	
};


#endif // INSTRUMENT_SUPPORT_INSTROSPECTION_DL_CODE_ADDRESS_INFO_BASE_HPP
