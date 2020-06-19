/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFMETADATA_HPP
#define CTFMETADATA_HPP

#include <set>
#include <string>
#include <cstdint>

#include "lowlevel/FatalErrorHandler.hpp"
#include "ctfapi/CTFEvent.hpp"

namespace CTFAPI {
	class CTFMetadata {
	private:
		static const char *meta_header;
		static const char *meta_typedefs;
		static const char *meta_trace;
		static const char *meta_env;
		static const char *meta_clock;
		static const char *meta_streamBounded;
		static const char *meta_streamUnbounded;

		static const char *meta_eventMetadataId;
		static const char *meta_eventMetadataStreamId;
		static const char *meta_eventMetadataFields;

		uint16_t totalCPUs;

		std::set<CTFEvent *> events;
		std::set<CTFContext *> contexes;

		void writeEventContextMetadata(FILE *f, CTFAPI::CTFEvent *event);
		void writeEventMetadata(FILE *f, CTFAPI::CTFEvent *event, int streamId);

	public:

		CTFMetadata() {};
		~CTFMetadata();

		CTFEvent *addEvent(CTFEvent *event)
		{
			auto ret = events.emplace(event);
			FatalErrorHandler::failIf(!ret.second, "Attempt to register a duplicate CTF Event with name ", event->getName());

			return event;
		}

		CTFContext *addContext(CTFContext *context)
		{
			auto ret = contexes.emplace(context);
			FatalErrorHandler::failIf(!ret.second, "Attempt to register a duplicate CTF Context");

			return context;
		}

		std::set<CTFEvent *> &getEvents()
		{
			return events;
		}

		void writeMetadataFile(std::string userPath);
	};
}

#endif // CTFMETADATA_HPP
