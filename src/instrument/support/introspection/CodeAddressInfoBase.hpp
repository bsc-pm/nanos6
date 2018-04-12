/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_INSTROSPECTION_CODE_ADDRESS_INFO_BASE_HPP
#define INSTRUMENT_SUPPORT_INSTROSPECTION_CODE_ADDRESS_INFO_BASE_HPP


#include <support/Objectified.hpp>

#include <map>
#include <string>
#include <vector>

#include <stdint.h>


class CodeAddressInfoBase {
public:
	typedef Objectified<uint32_t, ~0U> function_id_t;
	typedef Objectified<uint32_t, ~0U> source_location_id_t;
	
	
	struct InlineFrame {
		function_id_t _functionId;
		source_location_id_t _sourceLocationId;
		
		bool operator==(InlineFrame const &other) const
		{
			return (_functionId == other._functionId) && (_sourceLocationId == other._sourceLocationId);
		}
		
		bool operator!=(InlineFrame const &other) const
		{
			return (_functionId != other._functionId) || (_sourceLocationId != other._sourceLocationId);
		}
		
		bool operator<(InlineFrame const &other) const
		{
			if (_functionId < other._functionId) {
				return true;
			} else if (_functionId == other._functionId) {
				return (_sourceLocationId < other._sourceLocationId);
			} else {
				return false;
			}
		}
		
		bool operator>(InlineFrame const &other) const
		{
			if (_functionId > other._functionId) {
				return true;
			} else if (_functionId == other._functionId) {
				return (_sourceLocationId > other._sourceLocationId);
			} else {
				return false;
			}
		}
	};
	
	
	struct FrameNames {
		std::string const &_mangledFunction;
		std::string const &_function;
		std::string const &_sourceLocation;
		
		FrameNames(std::string const &mangledFunction, std::string const &function, std::string const &sourceLocation)
			: _mangledFunction(mangledFunction), _function(function), _sourceLocation(sourceLocation)
		{
		}
	};
	
	
	struct Entry {
		std::vector<InlineFrame> _inlinedFrames;
		void *_realAddress; // If generated from a return address, this one will (if possible) hold the actual call site
		
		bool empty() const
		{
			return _inlinedFrames.empty();
		}
		
		bool operator==(Entry const &other) const
		{
			if (_inlinedFrames.size() != other._inlinedFrames.size()) {
				return false;
			}
			
			auto it1 = _inlinedFrames.begin();
			auto it2 = other._inlinedFrames.begin();
			while (it1 != _inlinedFrames.end()) {
				if (*it1 != *it2) {
					return false;
				}
				
				it1++;
				it2++;
			}
			
			return true;
		}
		
		bool operator!=(Entry const &other) const
		{
			return !((*this) == other);
		}
		
		bool operator<(Entry const &other) const
		{
			auto it1 = _inlinedFrames.begin();
			auto it2 = other._inlinedFrames.begin();
			while ((it1 != _inlinedFrames.end()) && (it2 != other._inlinedFrames.end())) {
				if (*it1 < *it2) {
					return true;
				} else if (*it2 < *it1) {
					return false;
				}
				
				it1++;
				it2++;
			}
			
			return ((it1 == _inlinedFrames.end()) && (it2 != other._inlinedFrames.end()));
		}
		
		bool operator>(Entry const &other) const
		{
			auto it1 = _inlinedFrames.begin();
			auto it2 = other._inlinedFrames.begin();
			while ((it1 != _inlinedFrames.end()) && (it2 != other._inlinedFrames.end())) {
				if (*it1 > *it2) {
					return true;
				} else if (*it2 > *it1) {
					return false;
				}
				
				it1++;
				it2++;
			}
			
			return ((it1 != _inlinedFrames.end()) && (it2 == other._inlinedFrames.end()));
		}
	};
	
	
protected:
	static Entry _nullEntry;
	static std::string _unknownFunctionName;
	static std::string _unknownSourceLocation;
	
	static std::map<void *, Entry> _address2Entry;
	static std::map<void *, Entry> _returnAddress2Entry;
	
	static std::map<std::string, function_id_t> _functionName2Id;
	static std::map<std::string, source_location_id_t> _sourceLocation2Id;
	static std::vector<std::string> _mangledFunctionNames;
	static std::vector<std::string> _functionNames;
	static std::vector<std::string> _sourceLocations;
	
	static std::string demangleSymbol(std::string const &symbol);
	static std::string sourceToString(char const *source, int line, int column);
	
	static InlineFrame functionAndSourceToFrame(std::string const &mangledFunctionName, std::string const &functionName, std::string const &sourceLocation);
	
public:
	static inline std::string const &getMangledFunctionName(function_id_t functionId)
	{
		if (functionId != function_id_t()) {
			return _mangledFunctionNames[functionId];
		} else {
			return _unknownFunctionName;
		}
	}
	
	static inline std::string const &getFunctionName(function_id_t functionId)
	{
		if (functionId != function_id_t()) {
			return _functionNames[functionId];
		} else {
			return _unknownFunctionName;
		}
	}
	
	static inline std::string const &getSourceLocation(source_location_id_t sourceLocationId)
	{
		if (sourceLocationId != source_location_id_t()) {
			return _sourceLocations[sourceLocationId];
		} else {
			return _unknownSourceLocation;
		}
	}
	
	static inline FrameNames getFrameNames(InlineFrame const &frame)
	{
		return FrameNames(
			getMangledFunctionName(frame._functionId),
			getFunctionName(frame._functionId),
			getSourceLocation(frame._sourceLocationId)
		);
	}
	
};


#endif // INSTRUMENT_SUPPORT_INSTROSPECTION_CODE_ADDRESS_INFO_BASE_HPP
