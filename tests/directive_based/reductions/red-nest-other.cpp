/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether a nested reduction within a task with out or inout clauses is handled properly
 *
 */

#include "TestAnyProtocolProducer.hpp"


#define N 10

TestAnyProtocolProducer tap;

int main() {
	tap.registerNewTests(1);
	tap.begin();
	
	int x;
	
	#pragma oss task out(x)
	{
		#pragma oss task out(x)
		{
			x = 0;
			
			#pragma oss task weakreduction(+: x)
			{
				for (size_t i = 0; i < N; ++i)
				{
					#pragma oss task reduction(+: x)
					{
						x++;
					}
				}
			}
		}
	}
	
	#pragma oss task in(x)
	{
		std::ostringstream oss;
		oss << "Nested reduction within task with 'out' or 'inout' clauses properly computed";
		tap.evaluate(x == N, oss.str());
	}
	
	#pragma oss taskwait
	
	tap.end();
}
