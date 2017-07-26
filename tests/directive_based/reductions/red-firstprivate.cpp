/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>
#include <cassert>

#include "TestAnyProtocolProducer.hpp"


#define N 1000

TestAnyProtocolProducer tap;

int main()
{
	int n = N;
	
	assert(n > 0);
	
	tap.registerNewTests(1);
	tap.begin();
	
	int x = 0;
	#pragma oss task firstprivate(x)
	{
		for (int i = 0; i < n; ++i)
		{
			#pragma oss task reduction(+: x)
			{
				x++;
			}
		}
		
		#pragma oss task in(x)
		{
			std::ostringstream oss;
			oss << "Expected reduction result: x (" << x << ") == " << n;
			tap.evaluate(x == n, oss.str());
		}
	}
	#pragma oss taskwait
	
	tap.end();
	
	return 0;
}
