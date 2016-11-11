#include "TestAnyProtocolProducer.hpp"
#include "memory/directory/pages/MemoryPageSet.hpp"
#include "memory/directory/copies/CopySet.hpp"
#include "hardware/Machine.hpp"
#include "memory/directory/Directory.hpp"

static std::string pagesToString(MemoryPageSet &mySet) {
	std::ostringstream oss;
	
	oss << "Contents:" << std::endl;
	for (MemoryPageSet::iterator it = mySet.begin(); it != mySet.end(); it++) {
		oss << " " << it->getStartAddress() << " - " << it->getEndAddress() << " Size: " << it->getSize() << " Location:" << it->getLocation() << std::endl;
	}
	
	return oss.str();
}

int main(__attribute__((unused)) int argc, __attribute__((unused)) char **argv) {
	TestAnyProtocolProducer tap;	

	tap.registerNewTests(6);
	tap.begin();
	
	CopySet copies;
	MemoryPageSet pages;
	Machine::initialize();
	Directory::initialize();
		
	/*    BEGIN PAGES TESTS     */

	long pagesize = Machine::getMachine()->getPageSize();
	// Using 3 pages for test (for now)
	char *pagesTestData = (char *) malloc(sizeof(char) * pagesize * 3);
    
	DataAccessRange page1 = DataAccessRange( (void *)( (long) pagesTestData & ~(pagesize-1) ), pagesize );
	DataAccessRange page2 = DataAccessRange( page1.getEndAddress(), pagesize );	
	DataAccessRange page3 = DataAccessRange( page2.getEndAddress(), pagesize );	
	
	// 1
	tap.evaluate(pages.empty(), "the list is initially empty");
	tap.emitDiagnostic(pagesToString(pages));
	// 1	

	pages.insert(page1);
	
	// 2
	tap.evaluate(pages.size() == 1, "1 Element has been inserted");
	tap.emitDiagnostic(pagesToString(pages));
	// 2

	pages.insert(page3);

	// 3
	tap.evaluate(pages.size() == 2, "2 Elements have been inserted, there is no merge");
	tap.emitDiagnostic(pagesToString(pages));
	// 3

	pages.insert(page2);
	
	// 4
	tap.evaluate(pages.size() == 1, "The last elements has been inserted and all have merged");
	tap.emitDiagnostic(pagesToString(pages));
	// 4

	MemoryPageSet::iterator it = pages.find(page1.getStartAddress());
	
	// 5
	tap.evaluate(it != pages.end(), "The memory pages are found using the smallest starting address");
	tap.emitDiagnostic(pagesToString(pages));
	// 5

	// 6 
	tap.evaluate(it->getSize() == pagesize * 3, "The final object has the size of 3 pages");
	tap.emitDiagnostic(pagesToString(pages));
	// 6


	/*     END PAGES TEST    */

	
}
	
