/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAMUNBOUNDEDSHARED_HPP
#define CPUSTREAMUNBOUNDEDSHARED_HPP

#include "lowlevel/SpinLock.hpp"
#include "CTFStreamUnboundedPrivate.hpp"
#include "CTFStreamUnboundedPrivate.hpp"

namespace CTFAPI {
	class CTFStreamUnboundedShared : public CTFStreamUnboundedPrivate {
	private:
		SpinLock spinlock;

	public:
		CTFStreamUnboundedShared() : CTFStreamUnboundedPrivate() {}
		~CTFStreamUnboundedShared() {}

		void lock()
		{
			spinlock.lock();
		}

		void unlock()
		{
			spinlock.unlock();
		}
	};
}

#endif // CPUSTREAMUNBOUNDEDSHARED_HPP
