/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_KERNEL_METADATA_HPP
#define CTF_KERNEL_METADATA_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "CTFTypes.hpp"
#include "CTFMetadata.hpp"

namespace CTFAPI {
	class CTFKernelMetadata : public CTFMetadata {
	private:
		static const char *meta_header;
		static const char *meta_typedefs;
		static const char *meta_trace;
		static const char *meta_env;
		static const char *meta_clock;
		static const char *meta_stream;
		static const char *meta_event;

		static const char *defaultKernelDefsFileName;
		static const char *defaultEnabledEventsFileName;

		bool _enabled;
		std::string _kernelDefsBootId;
		ctf_kernel_event_id_t _numberOfEvents;
		ctf_kernel_event_id_t _maxEventId;
		std::string _kernelVersion;
		std::string _kernelRelease;

		std::map<std::string, std::pair<ctf_kernel_event_id_t, std::string> > _idMap;
		std::vector<ctf_kernel_event_id_t> _enabledEventIds;
		std::vector<std::string> _enabledEventNames;
		std::vector<ctf_kernel_event_size_t> _eventSizes;

		void checkBootId();
		bool getSystemInformation();
		bool loadKernelDefsFile(const char *file);
		bool loadEnabledEvents(const char *file);
	public:
		CTFKernelMetadata();
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
			auto const& entry = _idMap.at(eventName);
			return entry.first;
		}

		void writeMetadataFile(std::string kernelPath);
		void copyKernelDefinitionsFile(std::string basePath);
	};
}

#endif // CTF_KERNEL_METADATA_HPP
