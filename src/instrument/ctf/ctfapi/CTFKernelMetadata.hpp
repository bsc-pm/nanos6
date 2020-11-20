/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_KERNEL_METADATA_HPP
#define CTF_KERNEL_METADATA_HPP

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "CTFMetadata.hpp"
#include "CTFTypes.hpp"
#include "support/config/ConfigVariable.hpp"

namespace CTFAPI {
	class CTFKernelMetadata : public CTFMetadata {
	private:
		typedef std::map<std::string, std::pair<ctf_kernel_event_id_t, std::string> > kernel_event_map_t;

		static const char *meta_header;
		static const char *meta_typedefs;
		static const char *meta_trace;
		static const char *meta_env;
		static const char *meta_clock;
		static const char *meta_stream;
		static const char *meta_event;

		static ConfigVariableList<std::string> _kernelEventPresets;
		static ConfigVariable<std::string> _kernelEventFile;
		static ConfigVariableList<std::string> _kernelExcludedEvents;
		static std::map<std::string, std::vector<std::string> > _kernelEventPresetMap;

		bool _enabled;
		bool _enableSyscalls;

		std::string _kernelDefsBootId;
		ctf_kernel_event_id_t _numberOfEvents;
		ctf_kernel_event_id_t _maxEventId;
		std::string _kernelVersion;
		std::string _kernelRelease;

		kernel_event_map_t _kernelEventMap;
		std::vector<ctf_kernel_event_id_t> _enabledEventIds;
		std::set<std::string> _enabledEventNames;
		std::vector<ctf_kernel_event_size_t> _eventSizes;

		bool getSystemInformation();
		void loadKernelDefsFile(const char *file);
		bool loadEnabledEvents();
		bool addEventsInFile();
		bool addEventsInPresets();
		void addPatternDependentEvents();
		void excludeEvents();
		void translateEvents();
	public:

		CTFKernelMetadata()
			: _enabled(false), _enableSyscalls(false),  _maxEventId(-1)
		{
		}

		~CTFKernelMetadata()
		{
		};

		std::vector<ctf_kernel_event_id_t>& getEnabledEvents()
		{
			return _enabledEventIds;
		}

		std::vector<ctf_kernel_event_size_t>& getEventSizes()
		{
			return _eventSizes;
		}

		bool enabled()
		{
			return _enabled;
		}

		ctf_kernel_event_id_t getEventIdByName(std::string eventName) const
		{
			auto const& entry = _kernelEventMap.at(eventName);
			return entry.first;
		}

		void writeMetadataFile(std::string kernelPath);
		void parseKernelEventDefinitions();
		void initialize();
	};
}

#endif // CTF_KERNEL_METADATA_HPP
