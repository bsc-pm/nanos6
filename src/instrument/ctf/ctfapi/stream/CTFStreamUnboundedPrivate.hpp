/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAMUNBOUNDEDPRIVATE_HPP
#define CPUSTREAMUNBOUNDEDPRIVATE_HPP

#include <cassert>

#include "CTFStream.hpp"
#include "../context/CTFContextUnbounded.hpp"

namespace CTFAPI {
	class CTFStreamUnboundedPrivate : public CTFStream {
	public:
		CTFStreamUnboundedPrivate(size_t size,
					  ctf_cpu_id_t cpu,
					  std::string path)
			: CTFStream(size, cpu, path, 1)
		{
		}

		~CTFStreamUnboundedPrivate() {}

		virtual size_t getContextSize(void) const
		{
			assert(context != nullptr);
			return context->getSize();
		}

		void writeContext(void **buf);
	};
}

#endif // CPUSTREAMUNBOUNDEDPRIVATE_HPP
