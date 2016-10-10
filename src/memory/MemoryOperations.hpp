#ifndef MEMORY_OPS_HPP
#define MEMORY_OPS_HPP

#include <cstddef>
#include "hardware/Machine.hpp"

namespace MemoryOps {

	inline void *add(void *ptr, std::size_t bytes){
		return static_cast<void *>( static_cast<char *>( ptr ) + bytes );
	}

	inline void *sub(void *ptr, std::size_t bytes){
		return static_cast<void *>( static_cast<char *>( ptr ) - bytes );
	}

	inline std::ptrdiff_t distance(void *first, void *second){
		return static_cast<char *>( first ) - static_cast<char *>( second );
	}

	inline void *pageStart(void *address){
		return (void *) ( (long)address & ~(Machine::getMachine()->getPageSize()-1) );
	}

	inline unsigned int pageNumber(void *start, void *end){
		void *page = MemoryOps::pageStart(start);
		int bytes = MemoryOps::distance(page, end);
	
		return bytes / Machine::getMachine()->getPageSize();
	}
}

#endif //MEMORY_OPS_HPP
