#include "GenericIds.hpp"


namespace Instrument {
	namespace GenericIds {
		// 0 is reserved for the leader thread
		std::atomic<thread_id_t::inner_type_t> _nextThreadId(1);
	}
}

