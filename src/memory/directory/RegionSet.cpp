#include "RegionSet.hpp"

#include <vector>
#include <set>
#include <algorithm>

#include "hardware/Machine.hpp"
#include "memory/Region.hpp"


RegionSet::iterator RegionSet::begin(){
    return _set.begin();
}

RegionSet::iterator RegionSet::end(){
    return _set.end();
}

RegionSet::iterator RegionSet::find(void *address){
    return _set.find(address);
}

vector<MemoryRegion> RegionSet::insert(TaskMemoryData data){
	std::vector<RegionSet::iterator> regions;	

	//Structures to extract the pages of the task data
	std::vector<void *> p();
	std::vector<int> s();
	
	long pagesize = Machine::getMachine()->getPageSize();

	for(int i = 0; i < data.regions; i++){
		insertRegionPages(data.bases[i], data.sizes[i], pagesize, p);
	}

	void **pages = &p[0];
	int *status = &s[0];
	int npages = p.size();

	move_pages(0, npages, pages, NULL, status, 0); //Find the location of pages

	initial = p[0];
	psize = pagesize;
	node = s[0];

	/*
		Insert located pages to the set
		Pages are sorted and same size, so we can find which are adjacent and merge
	*/
	for(int i = 1; i < npages, i++){
		if(p[i] != Region::add(initial, psize) /* Pages are not touching */ || s[i] != node /* Different node */){
	
			std::pair<RegionSet::iterator, bool> response =  _set.insert(MemoryRegion(initial, psize, Machine::getMachine()->getMemoryNode(node)));
			responses.push_back(response->first*);

			//update next region values
			initial = p[i];
			psize = pagesize;
			node = s[i];
	
		} else { // Pages are touching and in the same node
			psize += pagesize;
		}

	}

	std::pair<RegionSet::iterator, bool> response = _set.insert(MemoryRegion(initial, psize, Machine::getMachine()->getMemoryNode(node)));
	responses.push_back(response->first*);
	
	return response;
}   
/* Insert the pages that the region touches if they are not already present */
static void insertRegionPages(void *baseAddress, int size, long pagesize, std::vector<void *> p, std::vector<int> s){
	void *page = (void *)( (long) baseAddress & ~(pagesize-1) );
	size += Region::distance(baseAddress, page);
	
	int npages = size / pagesize;

	//Check if element is already in queue before pushing
	if(std::find(p.begin(), p.end(), page) == p.end()) {
		p.push_back(page);
		s.push_back(0);		
	}

	for( int i = 1; i < npages; i++){
		page += pagesize;
		if(std::find(p.begin(), p.end(), page) == p.end()){
			p.push_back(page);
			s.push_back(0);
		}
	}

	std::sort(p.begin(), p.end());
}

void RegionSet::insert(TaskMemoryData data){
	//std::vector<void *> p();
	//std::vector<int> s();
	//long pagesize = Machine::getMachine()->getPageSize();

	//for(int i = 0; i < data.regions; i++){
	//	//insertRegionPages(data.bases[i], data.sizes[i], pagesize, p, s);
	//}

	//void **pages = (void **) &p[0];
	//int *status = (int *) &s[0];
	//int npages = p.size();

	//move_pages(0, npages, pages, NULL, status, 0); //Find the location of pages

	//page = p[0];
	//pst = s[0];
	//for(int i = 1; i < npages, i++){
	//	if(p[i] ==  
	//}
}   

