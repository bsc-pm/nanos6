/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "CodeAddressInfo.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include <unistd.h>
#include <sys/types.h>


std::map<void *, CodeAddressInfo::MemoryMapSegment> CodeAddressInfo::_executableMemoryMap;


void CodeAddressInfo::init()
{
	pid_t pid = getpid();
	
	std::string mapsFilename;
	{
		std::ostringstream oss;
		oss << "/proc/" << pid << "/maps";
		mapsFilename = oss.str();
	}
	std::ifstream mapsFile(mapsFilename.c_str());
	
	if (!mapsFile.is_open()) {
		std::cerr << "Warning: cannot get the memory map of the process from '" << mapsFilename << "'" << std::endl;
		return;
	}
	
	std::istringstream splitter;
	std::string field;
	std::istringstream hexDecoder;
	hexDecoder.setf(std::ios::hex, std::ios::basefield);
	while (!mapsFile.eof() && !mapsFile.bad()) {
		std::string line;
		std::getline(mapsFile, line);
		
		if (mapsFile.eof()) {
			break;
		} else if (mapsFile.bad()) {
			std::cerr << "Warning: error getting the memory map of the process from '" << mapsFilename << "'" << std::endl;
			break;
		}
		
		splitter.clear();
		splitter.str(line);
		
		// Memory start address
		size_t baseAddress;
		std::getline(splitter, field, '-');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> baseAddress;
		
		MemoryMapSegment &memoryMapSegment = _executableMemoryMap[(void *) baseAddress];
		
		// Memory end address + 1
		std::getline(splitter, field, ' ');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> memoryMapSegment._length;
		memoryMapSegment._length -= baseAddress;
		
		// Permissions
		std::getline(splitter, field, ' ');
		
		// Offset
		std::getline(splitter, field, ' ');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> memoryMapSegment._offset;
		
		// Device
		std::getline(splitter, field, ' ');
		
		// Inode
		long inode;
		splitter >> inode;
		
		// Path (if any)
		std::string path;
		std::getline(splitter, path);
		{
			size_t beginningOfPath = path.find_first_not_of(' ');
			if (beginningOfPath != std::string::npos) {
				path = path.substr(beginningOfPath);
				if (!path.empty() && (path[0] != '[')) {
					memoryMapSegment._filename = std::move(path);
				}
			}
		}
	}
	
	mapsFile.close();
}


void CodeAddressInfo::shutdown()
{
}


CodeAddressInfo::Entry const &CodeAddressInfo::resolveAddress(void* address)
{
	{
		auto it = _address2Entry.find(address);
		if (it != _address2Entry.end()) {
			return it->second;
		}
	}
	
	auto it = _executableMemoryMap.upper_bound(address);
	if (it == _executableMemoryMap.begin()) {
		// The address cannot be resolved
		return _nullEntry;
	}
	it--;
	
	MemoryMapSegment const &memoryMapSegment = it->second;
	
	if (memoryMapSegment._filename.empty()) {
		return _nullEntry;
	}
	
	Entry entry;
	
	size_t relativeAddress = (size_t)address - (size_t)it->first;
	
	std::ostringstream addr2lineCommandLine;
	addr2lineCommandLine << "addr2line -i -f -C -e " << memoryMapSegment._filename << " " << std::hex << relativeAddress;
	
	FILE *addr2lineOutput = popen(addr2lineCommandLine.str().c_str(), "r");
	if (addr2lineOutput == NULL) {
		perror("Error executing addr2line");
		exit(1);
	}
	
	char buffer[8192];
	buffer[8191] = 0;
	size_t length = fread(buffer, 1, 8191, addr2lineOutput);
	std::string cpp_buffer(buffer, length);
	pclose(addr2lineOutput);
	
	std::istringstream output(cpp_buffer);
	std::string function;
	std::string sourceLine;
	std::getline(output, function);
	std::getline(output, sourceLine);
	
	while (!output.eof()) {
		if ((function != "??") && (sourceLine != "??:0") && (sourceLine != "??:?")) {
			// Add the current function and source location
			InlineFrame currentFrame = functionAndSourceToFrame(function, sourceLine);
			entry._inlinedFrames.push_back(currentFrame);
		}
		
		std::getline(output, function);
		std::getline(output, sourceLine);
	}
	
	_address2Entry[address] = std::move(entry);
	return _address2Entry[address];
}

