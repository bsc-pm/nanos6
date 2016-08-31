#include "TestAnyProtocolProducer.hpp"
#include "memory/Directory.hpp"
#include "memory/Directory.cpp"

#include <algorithm>
#include <sstream>
#include <string>

static std::string contentsToString(Directory &dir){
	std::ostringstream oss;
	
	oss << "Contents:" << std::endl;
	for(Directory::iterator it = dir.begin(); it != dir.end(); ++it){
		oss << *it << std::endl;
	}

	return oss.str();
}

int main(__attribute__((unused)) int argc, __attribute__((unused)) char **argv){
	TestAnyProtocolProducer tap;
	
	tap.registerNewTests(11);
	tap.begin();

	Directory dir;
	
	void *ptr1;
	void *ptr2;
		
	ptr1 = (void *) new int;

	ptr2 = (void *) new float[5][20];
	
	int *dim1 = new int[1];
	dim1[0] = sizeof(int);

	int *dim2 = new int[2];
 	dim2[0] = 5;
	dim2[1] = 20 * sizeof(float);
	
	if(ptr1 > ptr2){
		void *tmp = ptr1;
		int *tmpdim = dim1;
		
		ptr2 = ptr1;
		dim2 = dim1;

		ptr1 = tmp;
		dim1 = tmpdim;
	}
		
	
	dir.insert(ptr1, 1, dim1);
	dir.insert(ptr2, 2, dim2);
	
	//1
	tap.evaluate(dir.empty(), "The directory is initially empty");
	tap.emitDiagnostic(contentsToString(dir));
	
	//2 
	tap.evaluate( (dir.size() == 2 ), "There are two elements in the directory" );
	tap.emitDiagnostic(contentsToString(dir));

	Directory::iterator it = dir.begin();
	
	//3
	tap.evaluate( (it != dir.end() ), "Elements have been inserted on the directory" );
	tap.emitDiagnostic(contentsToString(dir));
	
	//4
	tap.evaluate( (it->getBaseAddress() == ptr1), "First element in the directory is smaller pointer" );		
	tap.emitDiagnostic(contentsToString(dir));

	it++;
	
	//5
	tap.evaluate( (it != dir.end() ), "Second element has been inserted to directory" );
	tap.emitDiagnostic(contentsToString(dir));	

	//6
	tap.evaluate( (it->getBaseAddress() == ptr2), "Second element of the directory is the larger pointer" );
	tap.emitDiagnostic(contentsToString(dir));

	it++;

	//7
	tap.evaluate( (it == dir.end() ), "No more elements have been inserted" );
	tap.emitDiagnostic(contentsToString(dir));

	it = dir.find(ptr2);
	
	//8
	tap.evaluate( (it->getBaseAddress() == ptr2) , "find() return the pointer asked for" );
	tap.emitDiagnostic(contentsToString(dir));
	
	dir.erase(it);
	
	//9
	tap.evaluate( dir.size() == 1, "erase() removes an element" );
	tap.emitDiagnostic(contentsToString(dir));

	it = dir.find(ptr2);

	//10
	tap.evaluate( it == dir.end(), "ptr2 is no longer present on the system" );
	tap.emitDiagnostic(contentsToString(dir));		

	it = dir.find(ptr1);
	dir.erase(it);
	
	//11
	tap.evaluate( dir.empty(), "directory is empty after erasing all elements" );
	tap.emitDiagnostic(contentsToString(dir));

	return 0;
}	
