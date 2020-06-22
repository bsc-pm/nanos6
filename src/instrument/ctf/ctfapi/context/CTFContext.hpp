/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXT_HPP
#define CTFCONTEXT_HPP

#include <string>

#include "../CTFTypes.hpp"

namespace CTFAPI {

	class CTFContext {
	protected:
		size_t size;
		std::string dataStructuresMetadata;

		CTFContext() : size(0) {}
	public:
		virtual ~CTFContext() {}

		inline const char *getDataStructuresMetadata() const
		{
			return dataStructuresMetadata.c_str();
		}
	};
}

#endif // CTFCONTEXT_HPP
