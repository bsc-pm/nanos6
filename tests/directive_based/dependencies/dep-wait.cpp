/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>
#include <unistd.h>
#include <cassert>
#include <string>

#include <Atomic.hpp>
#include "TestAnyProtocolProducer.hpp"

#define DEPTH 10000

TestAnyProtocolProducer tap;

Atomic<size_t> counter;

void task(int depth, int n)
{
	if (depth == n) {
		++counter;
	} else {
		#pragma oss task inout(counter) wait
		{
			#pragma oss task
			task(depth + 1, n);
		}
		
		#pragma oss task inout(counter)
		{
			size_t count = ++counter;
			if (depth != n - count + 1) {
				tap.evaluate(false, "The intermediate result of the program is correct");
				tap.end();
				exit(0);
			}
		}
	}
}

int main()
{
	int n = DEPTH;
	counter = 0;
	
	tap.registerNewTests(1);
	tap.begin();
	
	#pragma oss task
	task(0, n);
	
	#pragma oss taskwait
	
	tap.end();
	
	tap.evaluate(true, "The final result of the program is correct");
	
	return 0;
}
