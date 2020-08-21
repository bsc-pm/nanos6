/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAMUNBOUNDEDPRIVATE_HPP
#define CPUSTREAMUNBOUNDEDPRIVATE_HPP

#include <cassert>

#include "CTFStream.hpp"
#include "instrument/ctf/ctfapi/context/CTFStreamContextUnbounded.hpp"

namespace CTFAPI {
	class CTFStreamUnboundedPrivate : public CTFStream {
	protected:
		CTFStreamContextUnbounded *_context;
	public:
		CTFStreamUnboundedPrivate(size_t size,
					  ctf_cpu_id_t cpu,
					  std::string path)
			: CTFStream(size, cpu, path, CTFStreamUnboundedId)
		{
		}

		~CTFStreamUnboundedPrivate() {}

		inline void addContext(CTFStreamContextUnbounded *context)
		{
			_context = context;
		}

		size_t getContextSize() const override
		{
			assert(_context != nullptr);
			return _context->getSize();
		}

		inline void writeContext(void **buf) override
		{
			assert(_context != nullptr);
			_context->writeContext(buf);
		}
	};
}

#endif // CPUSTREAMUNBOUNDEDPRIVATE_HPP
