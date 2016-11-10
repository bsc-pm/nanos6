#include "MemoryPageSet.hpp"

#include "hardware/Machine.hpp"

#include <numaif.h>

MemoryPageSet::MemoryPageSet(): BaseType(){}


MemoryPageSet::iterator MemoryPageSet::find(void *address){
    return BaseType::find( DataAccessRange( address, address ) );
}

MemoryPageSet::iterator MemoryPageSet::insert(DataAccessRange range){
	long pagesize = Machine::getMachine()->getPageSize();
	void *page = (void *)( (long) range.getStartAddress() & ~(pagesize-1) );
	size_t size = static_cast<char *>( range.getEndAddress() ) - static_cast<char *>(page);
	
	int npages = size / pagesize;

	void * pages[npages];
	int status[npages]; 

	pages[0] = page;
	for(int i = 1; i < npages; i++){
		page += pagesize;
		pages[i] = page;
	}

	move_pages(0, npages, pages, NULL, status, 0);
	
	for(int i = 0; i < npages; i++){
		if(BaseType::find(DataAccessRange( pages[i], pages[i] )) != BaseType::end()){
			MemoryPageObject *obj = new MemoryPageObject(pages[i], pagesize, status[i]);
			BaseType::insert(*obj);
		}
	}
	
}
