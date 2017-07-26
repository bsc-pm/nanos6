/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef RUNTIME_INFO_HPP
#define RUNTIME_INFO_HPP


#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <string.h>

#include "lowlevel/SpinLock.hpp"

#include "api/nanos6/runtime-info.h"




class RuntimeInfo {
private:
	static SpinLock _lock;
	static std::vector<nanos6_runtime_info_entry_t> _contents;
	
	template <typename T, bool INT_CONVERTIBLE, bool DOUBLE_CONVERTIBLE, bool STRING_CONVERTIBLE, bool INTEGER, bool DOUBLE, bool C_STRING>
	struct EntryValueSetter {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value);
	};
	
	template <typename T, bool INT_CONVERTIBLE, bool DOUBLE_CONVERTIBLE, bool STRING_CONVERTIBLE>
	struct EntryValueSetter<T, INT_CONVERTIBLE, DOUBLE_CONVERTIBLE, STRING_CONVERTIBLE, true, false, false> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_integer_runtime_info_entry;
			entry.integer = value;
		}
	};
	
	template <typename T, bool INT_CONVERTIBLE, bool DOUBLE_CONVERTIBLE, bool STRING_CONVERTIBLE>
	struct EntryValueSetter<T, INT_CONVERTIBLE, DOUBLE_CONVERTIBLE, STRING_CONVERTIBLE, false, true, false> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_real_runtime_info_entry;
			entry.real = value;
		}
	};
	
	template <typename T, bool INT_CONVERTIBLE, bool STRING_CONVERTIBLE, bool DOUBLE>
	struct EntryValueSetter<T, INT_CONVERTIBLE, true, STRING_CONVERTIBLE, false, DOUBLE, false> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_real_runtime_info_entry;
			entry.real = value;
		}
	};
	
	template <typename T, bool INTEGER, bool DOUBLE>
	struct EntryValueSetter<T, false, false, true, INTEGER, DOUBLE, false> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_text_runtime_info_entry;
			entry.text = strdup( std::string(value).c_str() );
		}
	};
	
	template <typename T, bool STRING_CONVERTIBLE, bool INTEGER, bool DOUBLE>
	struct EntryValueSetter<T, true, false, STRING_CONVERTIBLE, INTEGER, DOUBLE, false> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_integer_runtime_info_entry;
			entry.integer = value;
		}
	};
	
	template <typename T, bool STRING_CONVERTIBLE>
	struct EntryValueSetter<T, false, false, STRING_CONVERTIBLE, false, false, true> {
		static void setEntryValue(nanos6_runtime_info_entry_t &entry, T const &value)
		{
			entry.type = nanos6_text_runtime_info_entry;
			entry.text = strdup(value);
		}
	};
	
	
public:
	template <typename T>
	static void addEntry(std::string const &name, std::string const &description, T const &value, std::string const &units = "")
	{
		_lock.lock();
		
		_contents.emplace_back();
		nanos6_runtime_info_entry_t &entry = _contents.back();
		
		entry.name = strdup(name.c_str());
		entry.description = strdup(description.c_str());
		
		EntryValueSetter<T,
			std::is_convertible<typename std::remove_reference<T>::type, long>::value,
			std::is_convertible<typename std::remove_reference<T>::type, double>::value,
			std::is_convertible<typename std::remove_reference<T>::type, std::string>::value, 
			std::is_integral<typename std::remove_reference<T>::type>::value,
			std::is_floating_point<typename std::remove_reference<T>::type>::value,
			std::is_same< typename std::remove_cv< typename std::remove_reference<T>::type >::type, char *>::value
		>::setEntryValue(entry, value);
		
		if (!units.empty()) {
			entry.units = strdup(units.c_str());
		} else {
			entry.units = "";
		}
		
		_lock.unlock();
	}
	
	
	template <typename ITERATOR_T>
	static void addListEntry(std::string const &name, std::string const &description, ITERATOR_T begin, ITERATOR_T end, std::string const &units = "")
	{
		std::ostringstream oss;
		
		oss << "{";
		for (auto it = begin; it != end; it++) {
			if (it != begin) {
				oss << ", ";
			}
			oss << *it;
		}
		oss << "}";
		
		addEntry(name, description, oss.str(), units);
	}
	
	
	static size_t size()
	{
		_lock.lock();
		size_t result = _contents.size();
		_lock.unlock();
		
		return result;
	}
	
	
	static void getEntryContents(size_t index, nanos6_runtime_info_entry_t *entry)
	{
		_lock.lock();
		*entry = _contents[index];
		_lock.unlock();
	}
};


#endif // RUNTIME_INFO_HPP
