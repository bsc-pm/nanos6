/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>
#include <bitset>
#include <ostream>

#include "DataAccess.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>

static inline bool matchAll(access_flags_t value, access_flags_t mask)
{
	return ((value & mask) == mask);
}

static inline access_flags_t calculatePropagationFlags(access_flags_t flagsForNext, PropagationDestination destination)
{
	access_flags_t flagsAfterPropagation = ACCESS_NONE;

	if (flagsForNext) {
		// Set the flags that we need to be reset
		if (destination == NEXT) {
			if (flagsForNext & ACCESS_READ_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_READ_SATISFIED;
			if (flagsForNext & ACCESS_WRITE_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_WRITE_SATISFIED;
		} else if (destination == CHILD) {
			if (flagsForNext & ACCESS_READ_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_READ_SATISFIED;
			if (flagsForNext & ACCESS_WRITE_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_WRITE_SATISFIED;
		} else if (destination == PARENT) {
			if (flagsForNext & ACCESS_CHILDS_FINISHED)
				flagsAfterPropagation |= ACCESS_NEXT_WRITE_SATISFIED;
			if (flagsForNext & ACCESS_EARLY_READ)
				flagsAfterPropagation |= ACCESS_NEXT_READ_SATISFIED;
		}
	}

	return flagsAfterPropagation;
}

static inline bool calculateDisposing(access_flags_t flags, access_flags_t oldFlags, bool reduction = false)
{
	access_flags_t allFlags = (flags | oldFlags);
	access_flags_t disposeFlags = ACCESS_WRITE_SATISFIED | ACCESS_READ_SATISFIED | ACCESS_UNREGISTERED;

	if (allFlags & ACCESS_HASCHILD) {
		disposeFlags |= (ACCESS_CHILD_READ_SATISFIED | ACCESS_CHILD_WRITE_SATISFIED | ACCESS_CHILDS_FINISHED | ACCESS_EARLY_READ);
	}

	if (allFlags & ACCESS_HASNEXT) {
		disposeFlags |= (ACCESS_NEXT_READ_SATISFIED | ACCESS_NEXT_WRITE_SATISFIED);
	} else if (allFlags & ACCESS_NEXTISPARENT) {
		disposeFlags |= (ACCESS_PARENT_DONE | ACCESS_NEXT_READ_SATISFIED | ACCESS_NEXT_WRITE_SATISFIED);
	} else {
		disposeFlags |= ACCESS_PARENT_DONE;
	}

	if (reduction)
		disposeFlags |= ACCESS_REDUCTION_COMBINED;

	return matchAll(allFlags, disposeFlags);
}

DataAccessMessage DataAccess::inAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	bool toNextOnly,
	bool weak)
{
	access_flags_t allFlags = flags | oldFlags;
	DataAccessMessage message;
	PropagationDestination destination = NONE;
	message.from = this;

	// Read accesses always have to propagate read satisfiability.
	if (flags & ACCESS_READ_SATISFIED) {
		message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_WRITE_SATISFIED))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		message.schedule = !weak;
	}

	// We can only propagate write satisfiability if the access is finished or
	// it is a weak access and has a child (and hence has to propagate to children)
	if (flags & ACCESS_WRITE_SATISFIED) {
		if (toNextOnly) {
			if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_READ_SATISFIED))
				message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		} else {
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		}
	}

	if (flags & ACCESS_UNREGISTERED) {
		if (matchAll(allFlags, ACCESS_CHILDS_FINISHED | ACCESS_READ_SATISFIED)) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
		}
	}

	if (flags & ACCESS_CHILDS_FINISHED) {
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_READ_SATISFIED)) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
		}
	}

	if (flags & ACCESS_NEXTISPARENT) {
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_WRITE_SATISFIED | ACCESS_READ_SATISFIED))
			message.flagsForNext |= ACCESS_CHILDS_FINISHED;

		if (allFlags & ACCESS_READ_SATISFIED)
			message.flagsForNext |= ACCESS_EARLY_READ;

		destination = PARENT;
	}

	// If we register a child, we must propagate for sure the READ satisifiability,
	// and even the WRITE if the current access is a weakin
	if (flags & ACCESS_HASCHILD) {
		message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));

		destination = CHILD;
	}

	// If we register a successor, we must propagate the READ, and the WRITE if we have finished and have no childs
	if (flags & ACCESS_HASNEXT) {
		message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_READ_SATISFIED))
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));

		destination = NEXT;
	}

	if (destination == NONE && message.flagsForNext) {
		if (allFlags & ACCESS_HASCHILD && !(allFlags & ACCESS_CHILDS_FINISHED) && !toNextOnly) {
			message.to = _child.load(std::memory_order_relaxed);
			destination = CHILD;
		} else if ((allFlags & ACCESS_HASNEXT) && toNextOnly) {
			message.to = _successor.load(std::memory_order_relaxed);
			destination = NEXT;
		} else if ((allFlags & ACCESS_NEXTISPARENT) && toNextOnly) {
			// We have to "translate" the flags for the parent
			access_flags_t forParent = ACCESS_NONE;

			if (message.flagsForNext & ACCESS_WRITE_SATISFIED)
				forParent |= ACCESS_CHILDS_FINISHED;
			if (message.flagsForNext & ACCESS_READ_SATISFIED)
				forParent |= ACCESS_EARLY_READ;

			message.flagsForNext = forParent;

			message.to = _successor.load(std::memory_order_relaxed);
			destination = PARENT;
		} else {
			message.flagsForNext = ACCESS_NONE;
		}
	} else if (message.flagsForNext) {
		if (destination == CHILD && !toNextOnly)
			message.to = _child.load(std::memory_order_relaxed);
		else if ((destination == PARENT || destination == NEXT) && toNextOnly)
			message.to = _successor.load(std::memory_order_relaxed);
	}

	message.flagsAfterPropagation = calculatePropagationFlags(message.flagsForNext, destination);

	return message;
}

void DataAccess::outAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	DataAccessMessage &message,
	bool weak)
{
	access_flags_t allFlags = flags | oldFlags;
	message.from = this;
	PropagationDestination destination = NONE;

	if (flags & ACCESS_READ_SATISFIED) {
		if (allFlags & ACCESS_WRITE_SATISFIED)
			message.schedule = !weak;

		if ((allFlags & ACCESS_UNREGISTERED) || (allFlags & ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_READ_SATISFIED;
	}

	if (flags & ACCESS_WRITE_SATISFIED) {
		if (allFlags & ACCESS_READ_SATISFIED)
			message.schedule = !weak;

		if ((allFlags & ACCESS_UNREGISTERED) || (allFlags & ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;
	}

	if (flags & ACCESS_UNREGISTERED) {
		if (allFlags & ACCESS_CHILDS_FINISHED) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		} else {
			if (matchAll(allFlags, ACCESS_EARLY_READ | ACCESS_HASNEXT)) {
				message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
				destination = NEXT;
			}
		}
	}

	if (flags & ACCESS_CHILDS_FINISHED) {
		if (allFlags & ACCESS_UNREGISTERED) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));

			if (!(oldFlags & ACCESS_EARLY_READ))
				message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		}
	}

	if (flags & ACCESS_EARLY_READ) {
		// We cannot have ACCESS_CHILDS_FINISHED, that means we lost a race.
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_HASNEXT) && !(oldFlags & ACCESS_CHILDS_FINISHED)) {
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));

			destination = NEXT;
		}
	}

	if (flags & ACCESS_NEXTISPARENT) {
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_WRITE_SATISFIED))
			message.flagsForNext |= (ACCESS_CHILDS_FINISHED | ACCESS_EARLY_READ);

		destination = PARENT;
	}

	if (flags & ACCESS_HASCHILD) {
		// Registering a child access
		message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
		message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));

		destination = CHILD;
	}

	if (flags & ACCESS_HASNEXT) {
		// What if I have children!
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED)) {
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
		} else if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_EARLY_READ)) {
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		}

		destination = NEXT;
	}

	if (destination != NONE && message.flagsForNext) {
		if (destination == NEXT || destination == PARENT)
			message.to = _successor.load(std::memory_order_relaxed);
		else
			message.to = _child.load(std::memory_order_relaxed);
	}

	if (message.to == nullptr && destination == NONE && message.flagsForNext) {
		if (allFlags & ACCESS_HASCHILD && !(allFlags & ACCESS_CHILDS_FINISHED)) {
			message.to = _child.load(std::memory_order_relaxed);
			destination = CHILD;
		} else if (allFlags & ACCESS_HASNEXT) {
			message.to = _successor.load(std::memory_order_relaxed);
			destination = NEXT;
		} else if (allFlags & ACCESS_NEXTISPARENT) {
			if (message.flagsForNext & ACCESS_WRITE_SATISFIED) {
				message.flagsForNext = (ACCESS_CHILDS_FINISHED | ACCESS_EARLY_READ);
			} else
				message.flagsForNext = ACCESS_NONE;

			destination = PARENT;
			message.to = _successor.load(std::memory_order_relaxed);
		} else
			message.flagsForNext = ACCESS_NONE;
	}

	message.flagsAfterPropagation = calculatePropagationFlags(message.flagsForNext, destination);
}

void DataAccess::inoutAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	DataAccessMessage &message,
	bool weak)
{
	this->outAutomata(flags, oldFlags, message, weak);
}

void DataAccess::reductionAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	DataAccessMessage &message)
{
	access_flags_t allFlags = flags | oldFlags;
	message.from = this;
	PropagationDestination destination = NONE;

	// No nested reductions, period.
	assert(!(allFlags & ACCESS_HASCHILD));

	if (flags & ACCESS_READ_SATISFIED) {
		if (allFlags & ACCESS_UNREGISTERED)
			message.flagsForNext |= ACCESS_READ_SATISFIED;
	}

	if (flags & ACCESS_WRITE_SATISFIED) {
		message.combine = true;
		message.flagsAfterPropagation |= ACCESS_REDUCTION_COMBINED;

		if (allFlags & ACCESS_UNREGISTERED)
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;
	}

	if (flags & ACCESS_UNREGISTERED) {
		if (allFlags & ACCESS_CHILDS_FINISHED) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		}
	}

	if (flags & ACCESS_CHILDS_FINISHED) {
		if (allFlags & ACCESS_UNREGISTERED) {
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
		}
	}

	if (flags & ACCESS_NEXTISPARENT) {
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED | ACCESS_WRITE_SATISFIED))
			message.flagsForNext |= (ACCESS_CHILDS_FINISHED | ACCESS_EARLY_READ);

		destination = PARENT;
	}

	if (flags & ACCESS_HASNEXT) {
		// What if I have children!
		if (matchAll(allFlags, ACCESS_UNREGISTERED | ACCESS_CHILDS_FINISHED)) {
			message.flagsForNext |= (allFlags & (ACCESS_READ_SATISFIED));
			message.flagsForNext |= (allFlags & (ACCESS_WRITE_SATISFIED));
		}

		destination = NEXT;
	}

	if (destination != NONE && message.flagsForNext) {
		if (destination == NEXT || destination == PARENT)
			message.to = _successor.load(std::memory_order_relaxed);
		else
			message.to = _child.load(std::memory_order_relaxed);
	}

	if (message.to == nullptr && message.flagsForNext) {
		if (allFlags & ACCESS_HASNEXT) {
			message.to = _successor.load(std::memory_order_relaxed);
			destination = NEXT;
		} else if (allFlags & ACCESS_NEXTISPARENT) {
			if (message.flagsForNext & ACCESS_WRITE_SATISFIED) {
				message.flagsForNext = (ACCESS_CHILDS_FINISHED | ACCESS_EARLY_READ);
			} else
				message.flagsForNext = ACCESS_NONE;

			destination = PARENT;
			message.to = _successor.load(std::memory_order_relaxed);
		} else
			message.flagsForNext = ACCESS_NONE;
	}

	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
}

bool DataAccess::applyPropagated(DataAccessMessage &message)
{
	if (message.flagsAfterPropagation == ACCESS_NONE)
		return false;

	bool isReduction = (getType() == REDUCTION_ACCESS_TYPE);
	access_flags_t oldFlags = _accessFlags.fetch_or(message.flagsAfterPropagation, std::memory_order_acquire);
	assert((oldFlags & message.flagsAfterPropagation) == ACCESS_NONE);
	assert(message.from == this);

	DataAccess *origin = (message.to == nullptr ? message.from : message.to);

	Instrument::automataMessage(origin->getInstrumentationId(), message.from->getInstrumentationId(), message.flagsAfterPropagation, oldFlags);

	bool dispose = calculateDisposing(message.flagsAfterPropagation, oldFlags, isReduction);

	return dispose;
}

bool DataAccess::apply(DataAccessMessage &message, mailbox_t &mailBox)
{
	if (message.flagsForNext == ACCESS_NONE)
		return false;

	bool isReduction = (getType() == REDUCTION_ACCESS_TYPE);
	DataAccessType type = getType();

	access_flags_t oldFlags = _accessFlags.fetch_or(message.flagsForNext);
	Instrument::automataMessage(message.from->getInstrumentationId(), message.to->getInstrumentationId(), message.flagsForNext, oldFlags);
	bool weak = (oldFlags & ACCESS_IS_WEAK);

	assert((oldFlags & message.flagsForNext) == ACCESS_NONE);
	assert(message.to == this);

	if (type == READ_ACCESS_TYPE) {
		DataAccessMessage toChild = inAutomata(message.flagsForNext, oldFlags, false, weak);
		DataAccessMessage toNext = inAutomata(message.flagsForNext, oldFlags, true, weak);

		if (toChild.to != nullptr && toChild.flagsForNext) {
			// Only one message can contain a dispose and schedule
			toNext.schedule = false;
			mailBox.push(toChild);
		}

		if ((toNext.to != nullptr && toNext.flagsForNext) || toNext.schedule) {
			mailBox.push(toNext);
		}
	} else if (type == WRITE_ACCESS_TYPE) {
		DataAccessMessage next;
		outAutomata(message.flagsForNext, oldFlags, next, weak);
		if (next.to != nullptr || next.schedule) {
			mailBox.push(next);
		}
	} else if (type == REDUCTION_ACCESS_TYPE) {
		DataAccessMessage next;
		reductionAutomata(message.flagsForNext, oldFlags, next);
		if (next.to != nullptr || next.combine) {
			mailBox.push(next);
		}
	} else {
		DataAccessMessage next;
		inoutAutomata(message.flagsForNext, oldFlags, next, weak);
		if (next.to != nullptr || next.schedule) {
			mailBox.push(next);
		}
	}

	// In case no message is returned, we still want to know if we need to delete this access.
	return calculateDisposing(message.flagsForNext, oldFlags, isReduction);
}

DataAccessMessage DataAccess::applySingle(access_flags_t flags, mailbox_t &mailBox)
{
	assert(mailBox.empty());

	DataAccessMessage message;
	message.flagsForNext = flags;
	message.to = this;
	message.from = this;

	__attribute__((unused)) bool dispose = this->apply(message, mailBox);
	assert(!dispose);

	// Max 1 message back
	assert(mailBox.size() <= 1);

	if (!mailBox.empty()) {
		message = mailBox.top();
		mailBox.pop();
	} else {
		message = DataAccessMessage();
	}

	return message;
}
