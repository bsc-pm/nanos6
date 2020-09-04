/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CTFEvent.hpp"

// Based on empirical observations, identifiers start at 1 to provide some
// protection against trace corruption. When a trace is corrupted, it is likely
// that zeros will be written when they should not, including the id field of an
// event. If identifiers start at 0, babeltrace will decode the corrupted area
// as the event with id 0 and will output misleading error messages. Instead, if
// starting at 1, babeltrace decoding will fail at this point.

// TODO use the ctf typedef here
uint8_t CTFAPI::CTFEvent::idCounter = 1;
