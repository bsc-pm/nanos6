/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <bitset>
#include <iostream>
#include <ostream>

#include "DataAccess.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>

#define PROPAGATE(a) (checkPropagation(flags, allFlags, a))

static inline bool matchAll(access_flags_t value, access_flags_t mask)
{
	return ((value & mask) == mask);
}

static inline bool checkPropagation(access_flags_t flags, access_flags_t allFlags, access_flags_t condition)
{
	return ((flags & condition) && matchAll(allFlags, condition));
}

// This could be a shift when flags are stable
static inline access_flags_t translateFlags(access_flags_t original)
{
	access_flags_t newFlags = ACCESS_NONE;

	if (original & ACCESS_READ_SATISFIED)
		newFlags |= ACCESS_CHILD_READ_DONE;

	if (original & ACCESS_WRITE_SATISFIED)
		newFlags |= ACCESS_CHILD_WRITE_DONE;

	if (original & ACCESS_CONCURRENT_SATISFIED)
		newFlags |= ACCESS_CHILD_CONCURRENT_DONE;

	if (original & ACCESS_COMMUTATIVE_SATISFIED)
		newFlags |= ACCESS_CHILD_COMMUTATIVE_DONE;

	assert(newFlags);
	return newFlags;
}

static inline access_flags_t calculatePropagationFlags(access_flags_t flagsForNext, PropagationDestination destination)
{
	access_flags_t flagsAfterPropagation = ACCESS_NONE;

	if (flagsForNext) {
		// Set the flags that we need to be set after delivering the message
		if (destination == NEXT) {
			if (flagsForNext & ACCESS_READ_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_READ_SATISFIED;
			if (flagsForNext & ACCESS_WRITE_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_WRITE_SATISFIED;
			if (flagsForNext & ACCESS_CONCURRENT_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_CONCURRENT_SATISFIED;
			if (flagsForNext & ACCESS_COMMUTATIVE_SATISFIED)
				flagsAfterPropagation |= ACCESS_NEXT_COMMUTATIVE_SATISFIED;
		} else if (destination == CHILD) {
			if (flagsForNext & ACCESS_READ_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_READ_SATISFIED;
			if (flagsForNext & ACCESS_WRITE_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_WRITE_SATISFIED;
			if (flagsForNext & ACCESS_CONCURRENT_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_CONCURRENT_SATISFIED;
			if (flagsForNext & ACCESS_COMMUTATIVE_SATISFIED)
				flagsAfterPropagation |= ACCESS_CHILD_COMMUTATIVE_SATISFIED;
		} else if (destination == PARENT) {
			if (flagsForNext & ACCESS_CHILD_WRITE_DONE)
				flagsAfterPropagation |= ACCESS_NEXT_WRITE_SATISFIED;
			if (flagsForNext & ACCESS_CHILD_READ_DONE)
				flagsAfterPropagation |= ACCESS_NEXT_READ_SATISFIED;
			if (flagsForNext & ACCESS_CHILD_CONCURRENT_DONE)
				flagsAfterPropagation |= ACCESS_NEXT_CONCURRENT_SATISFIED;
			if (flagsForNext & ACCESS_CHILD_COMMUTATIVE_DONE)
				flagsAfterPropagation |= ACCESS_NEXT_COMMUTATIVE_SATISFIED;
		}
	}

	return flagsAfterPropagation;
}

static inline bool calculateDisposing(access_flags_t flags, access_flags_t oldFlags, bool reduction = false)
{
	access_flags_t allFlags = (flags | oldFlags);
	access_flags_t disposeFlags = (ACCESS_WRITE_SATISFIED
		| ACCESS_READ_SATISFIED
		| ACCESS_CONCURRENT_SATISFIED
		| ACCESS_COMMUTATIVE_SATISFIED
		| ACCESS_UNREGISTERED);

	if (allFlags & ACCESS_HASCHILD) {
		disposeFlags |= (ACCESS_CHILD_READ_SATISFIED
			| ACCESS_CHILD_WRITE_SATISFIED
			| ACCESS_CHILD_CONCURRENT_SATISFIED
			| ACCESS_CHILD_COMMUTATIVE_SATISFIED
			| ACCESS_CHILD_WRITE_DONE
			| ACCESS_CHILD_READ_DONE
			| ACCESS_CHILD_CONCURRENT_DONE
			| ACCESS_CHILD_COMMUTATIVE_DONE);
	}

	if (allFlags & ACCESS_NEXTISPARENT) {
		disposeFlags |= (ACCESS_PARENT_DONE
			| ACCESS_NEXT_READ_SATISFIED
			| ACCESS_NEXT_WRITE_SATISFIED
			| ACCESS_NEXT_CONCURRENT_SATISFIED
			| ACCESS_NEXT_COMMUTATIVE_SATISFIED);
	} else if (allFlags & ACCESS_HASNEXT) {
		disposeFlags |= (ACCESS_NEXT_READ_SATISFIED
			| ACCESS_NEXT_WRITE_SATISFIED
			| ACCESS_NEXT_CONCURRENT_SATISFIED
			| ACCESS_NEXT_COMMUTATIVE_SATISFIED);
	} else {
		disposeFlags |= ACCESS_PARENT_DONE;
	}

	if (reduction)
		disposeFlags |= ACCESS_REDUCTION_COMBINED;

	return matchAll(allFlags, disposeFlags);
}

void DataAccess::readDestination(
	access_flags_t allFlags,
	DataAccessMessage &message,
	PropagationDestination &destination)
{
	if (destination != NONE && message.flagsForNext) {
		// A message is generated, safe to read the pointers
		if (destination == NEXT) {
			message.to = _successor.load(std::memory_order_relaxed);

			if (allFlags & ACCESS_NEXTISPARENT) {
				message.flagsForNext = translateFlags(message.flagsForNext);
				destination = PARENT;
			}
		} else {
			assert(destination == CHILD);
			message.to = _child.load(std::memory_order_relaxed);
		}
	} else {
		assert(!message.flagsForNext);
	}
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

	// This automata is called two times, one for the child message and another one for the next.
	// We handle this through two different sub-automatas for each destination

	if (flags & ACCESS_READ_SATISFIED)
		message.schedule = !weak;

	if (toNextOnly) {
		// Only messages that would go to successor or parent
		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_WRITE_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_WRITE_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_CONCURRENT_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_CONCURRENT_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_COMMUTATIVE_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_COMMUTATIVE_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		destination = NEXT;
	} else {
		// Only messages that would go to the child

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		destination = CHILD;
	}

	readDestination(allFlags, message, destination);
	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
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

	if (flags & ACCESS_WRITE_SATISFIED)
		message.schedule = !weak;

	if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_READ_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_READ_DONE)) {
		message.flagsForNext |= ACCESS_READ_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_WRITE_DONE)) {
		message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_CONCURRENT_DONE)) {
		message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_COMMUTATIVE_DONE)) {
		message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	readDestination(allFlags, message, destination);
	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
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

	if (flags & ACCESS_WRITE_SATISFIED) {
		message.combine = true;
		message.flagsAfterPropagation |= ACCESS_REDUCTION_COMBINED;
	}

	if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_READ_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_READ_DONE)) {
		message.flagsForNext |= ACCESS_READ_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_WRITE_DONE)) {
		message.flagsForNext |= ACCESS_WRITE_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_CONCURRENT_DONE)) {
		message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASCHILD)) {
		message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;
		assert(destination == NONE || destination == CHILD);
		destination = CHILD;
	}

	if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASNEXT | ACCESS_UNREGISTERED | ACCESS_CHILD_COMMUTATIVE_DONE)) {
		message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;
		assert(destination == NONE || destination == NEXT);
		destination = NEXT;
	}

	readDestination(allFlags, message, destination);
	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
}

DataAccessMessage DataAccess::concurrentAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	bool toNextOnly,
	bool weak)
{
	access_flags_t allFlags = flags | oldFlags;
	DataAccessMessage message;
	PropagationDestination destination = NONE;
	message.from = this;

	// This automata is called two times, one for the child message and another one for the next.
	// We handle this through two different sub-automatas for each destination

	if (flags & ACCESS_CONCURRENT_SATISFIED)
		message.schedule = !weak;

	if (toNextOnly) {
		// Only messages that would go to successor or parent
		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_WRITE_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_WRITE_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_READ_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_READ_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_COMMUTATIVE_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_COMMUTATIVE_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		destination = NEXT;
	} else {
		// Only messages that would go to the child

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		destination = CHILD;
	}

	readDestination(allFlags, message, destination);
	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
	return message;
}

DataAccessMessage DataAccess::commutativeAutomata(
	access_flags_t flags,
	access_flags_t oldFlags,
	bool toNextOnly,
	bool weak)
{
	access_flags_t allFlags = flags | oldFlags;
	DataAccessMessage message;
	PropagationDestination destination = NONE;
	message.from = this;

	// This automata is called two times, one for the child message and another one for the next.
	// We handle this through two different sub-automatas for each destination

	if (flags & ACCESS_COMMUTATIVE_SATISFIED)
		message.schedule = !weak;

	if (toNextOnly) {
		// Only messages that would go to successor or parent
		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_WRITE_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_WRITE_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_CONCURRENT_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_CONCURRENT_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_READ_SATISFIED | ACCESS_UNREGISTERED | ACCESS_CHILD_READ_DONE | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASNEXT))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		destination = NEXT;
	} else {
		// Only messages that would go to the child

		if (PROPAGATE(ACCESS_READ_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_READ_SATISFIED;

		if (PROPAGATE(ACCESS_WRITE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_WRITE_SATISFIED;

		if (PROPAGATE(ACCESS_CONCURRENT_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_CONCURRENT_SATISFIED;

		if (PROPAGATE(ACCESS_COMMUTATIVE_SATISFIED | ACCESS_HASCHILD))
			message.flagsForNext |= ACCESS_COMMUTATIVE_SATISFIED;

		destination = CHILD;
	}

	readDestination(allFlags, message, destination);
	message.flagsAfterPropagation |= calculatePropagationFlags(message.flagsForNext, destination);
	return message;
}

bool DataAccess::applyPropagated(DataAccessMessage &message)
{
	if (message.flagsAfterPropagation == ACCESS_NONE)
		return false;

	bool isReduction = (getType() == REDUCTION_ACCESS_TYPE);
	DataAccess *origin = (message.to == nullptr ? message.from : message.to);

	access_flags_t oldFlags = _accessFlags.fetch_add(message.flagsAfterPropagation, std::memory_order_acquire);
	Instrument::automataMessage(origin->getInstrumentationId(), message.from->getInstrumentationId(), message.flagsAfterPropagation, oldFlags);
	// No references to the access from here, as it could be deleted by another thread.
	// Any access without knowing for sure that a message will be generated is a use-after-free.
	assert((oldFlags & message.flagsAfterPropagation) == ACCESS_NONE);
	assert(message.from == this);

	bool dispose = calculateDisposing(message.flagsAfterPropagation, oldFlags, isReduction);

	return dispose;
}

bool DataAccess::apply(DataAccessMessage &message, mailbox_t &mailBox)
{
	if (message.flagsForNext == ACCESS_NONE)
		return false;

	bool isReduction = (getType() == REDUCTION_ACCESS_TYPE);
	DataAccessType type = getType();

	access_flags_t oldFlags = _accessFlags.fetch_add(message.flagsForNext, std::memory_order_acq_rel);
	Instrument::automataMessage(message.from->getInstrumentationId(), message.to->getInstrumentationId(), message.flagsForNext, oldFlags);
	// No references to the access from here, as it could be deleted by another thread.
	// Any access without knowing for sure that a message will be generated is a use-after-free.
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
	} else if (type == CONCURRENT_ACCESS_TYPE) {
		DataAccessMessage toChild = concurrentAutomata(message.flagsForNext, oldFlags, false, weak);
		DataAccessMessage toNext = concurrentAutomata(message.flagsForNext, oldFlags, true, weak);

		if (toChild.to != nullptr && toChild.flagsForNext) {
			// Only one message can contain a dispose and schedule
			toNext.schedule = false;
			mailBox.push(toChild);
		}

		if ((toNext.to != nullptr && toNext.flagsForNext) || toNext.schedule) {
			mailBox.push(toNext);
		}
	} else if (type == COMMUTATIVE_ACCESS_TYPE) {
		DataAccessMessage toChild = commutativeAutomata(message.flagsForNext, oldFlags, false, weak);
		DataAccessMessage toNext = commutativeAutomata(message.flagsForNext, oldFlags, true, weak);

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
