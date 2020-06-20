/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFEVENT_HPP
#define CTFEVENT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "context/CTFEventContext.hpp"

namespace CTFAPI {
	class CTFEvent {
	private:
		const uint8_t _id;
		const char *_name;
		const char *_metadataFields;
		bool _enabled;
		uint8_t _enabledContexes;

		std::vector<CTFEventContext *> eventContext;

		// TODO use CTF typedef to set the types
		static uint8_t idCounter;

	public:

		CTFEvent(const char *name, const char *metadataFields, uint8_t contexes = 0)
			: _id(idCounter++), _name(name), _metadataFields(metadataFields), _enabled(true),
			_enabledContexes(contexes)
		{
		}

		void writeContext(void **buf, ctf_stream_id_t streamId)
		{
			for (auto it = eventContext.begin(); it != eventContext.end(); it++)
				(*it)->writeContext(buf, streamId);
		}

		void addContext(CTFEventContext *context)
		{
			if (!context)
				return;

			eventContext.push_back(context);
		}

		size_t getContextSize(ctf_stream_id_t streamId) const
		{
			size_t contextSize = 0;
			for (auto it = eventContext.begin(); it != eventContext.end(); it++)
				contextSize += (*it)->getSize(streamId);
			return contextSize;
		}

		const char *getName() const
		{
			return _name;
		}

		bool isEnabled() const
		{
			return _enabled;
		}

		uint8_t getEnabledContexes() const
		{
			return _enabledContexes;
		}

		uint8_t getEventId() const
		{
			return _id;
		}

		const char *getMetadataFields() const
		{
			return _metadataFields;
		}

		std::vector<CTFEventContext *>& getContexes()
		{
			return eventContext;
		}

		bool operator< (const CTFEvent &event) const
		{
			return (_name < event.getName());
		}
	};
}


#endif // CTFEVENT_HPP
