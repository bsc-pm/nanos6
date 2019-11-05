/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LINT_TASK_ID_HPP
#define INSTRUMENT_LINT_TASK_ID_HPP


namespace Instrument {
	class task_id_t {
	public:
		typedef long int inner_type_t;

	private:
		inner_type_t _id;

	public:
		task_id_t(inner_type_t id)
		: _id(id)
		{
		}

		task_id_t()
		: _id(-1)
		{
		}

		operator inner_type_t() const
		{
			return _id;
		}
	};
}

#endif // INSTRUMENT_LINT_TASK_ID_HPP
