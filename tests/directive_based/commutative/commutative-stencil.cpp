/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <cassert>
#include <cstdio>
#include <sstream>

#include <string.h>
#include <unistd.h>

#include <Atomic.hpp>
#include "TestAnyProtocolProducer.hpp"


enum { BLOCK_SIZE = 1024 };
enum { NUM_BLOCKS = 10 };
enum { NUM_ITERATIONS = 8 };
enum { NUM_ROUNDS = 3 };


Atomic<int> expectedBlockValues[NUM_BLOCKS][NUM_BLOCKS];
int matrix[NUM_BLOCKS][NUM_BLOCKS][BLOCK_SIZE][BLOCK_SIZE];


TestAnyProtocolProducer tap;


static void init(bool initExpectedValues)
{
	for (int i = 0; i < NUM_BLOCKS; i++) {
		for (int j = 0; j < NUM_BLOCKS; j++) {
			if (initExpectedValues) {
				expectedBlockValues[i][j] = 0;
			}
			
			#pragma oss task out(matrix[i][j]) label(init)
			for (int ii = 0; ii < BLOCK_SIZE; ii++) {
				for (int jj = 0; jj < BLOCK_SIZE; jj++) {
					matrix[i][j][ii][jj] = 0;
				}
			}
		}
	}
}


static void blockUpdate(int i, int j)
{
	for (int ii = 0; ii < BLOCK_SIZE; ii++) {
		for (int jj = 0; jj < BLOCK_SIZE; jj++) {
			matrix[i][j][ii][jj]++;
		}
	}
}


static void cross_iteration(bool updateExpectedValues)
{
	#pragma oss task weakcommutative(([NUM_BLOCKS] matrix)) label(cross iteration)
	for (int i = 1; i < NUM_BLOCKS-1; i++) {
		for (int j = 1; j < NUM_BLOCKS-1; j++) {
			if (updateExpectedValues) {
				expectedBlockValues[i][j]++;
				expectedBlockValues[i-1][j]++;
				expectedBlockValues[i+1][j]++;
				expectedBlockValues[i][j-1]++;
				expectedBlockValues[i][j+1]++;
			}
			
			#pragma oss task \
				commutative(matrix[i][j]) \
				commutative(matrix[i-1][j]) \
				commutative(matrix[i+1][j]) \
				commutative(matrix[i][j-1]) \
				commutative(matrix[i][j+1]) \
				label(update cross)
			{
				blockUpdate(i, j);
				blockUpdate(i-1, j);
				blockUpdate(i+1, j);
				blockUpdate(i, j-1);
				blockUpdate(i, j+1);
			}
		}
	}
}


static void circumflex_iteration(bool updateExpectedValues)
{
	#pragma oss task weakcommutative(([NUM_BLOCKS] matrix)) label(circumflex iteration)
	for (int i = 1; i < NUM_BLOCKS-1; i++) {
		for (int j = 1; j < NUM_BLOCKS-1; j++) {
			if (updateExpectedValues) {
				expectedBlockValues[i][j]++;
				expectedBlockValues[i][j-1]++;
				expectedBlockValues[i][j+1]++;
			}
			
			#pragma oss task \
				commutative(matrix[i][j]) \
				commutative(matrix[i][j-1]) \
				commutative(matrix[i][j+1]) \
				label(update circumflex)
			{
				blockUpdate(i, j);
				blockUpdate(i, j-1);
				blockUpdate(i, j+1);
			}
		}
	}
}


static void spread_iteration(bool updateExpectedValues)
{
	#pragma oss task weakcommutative(([NUM_BLOCKS] matrix)) label(spread iteration)
	for (int i = 1; i < NUM_BLOCKS-1; i++) {
		for (int j = 1; j < NUM_BLOCKS-1; j++) {
			if (updateExpectedValues) {
				expectedBlockValues[i-1][j-1]++;
				expectedBlockValues[i+1][j+1]++;
			}
			
			#pragma oss task \
				commutative(matrix[i-1][j-1]) \
				commutative(matrix[i+1][j+1]) \
				label(update spread)
			{
				blockUpdate(i-1, j-1);
				blockUpdate(i+1, j+1);
			}
		}
	}
}


static void edges_iteration(bool updateExpectedValues)
{
	#pragma oss task weakcommutative(([NUM_BLOCKS] matrix)) label(edges iteration)
	for (int i = 1; i < NUM_BLOCKS-1; i++) {
		for (int j = 1; j < NUM_BLOCKS-1; j++) {
			if (updateExpectedValues) {
				expectedBlockValues[i-1][j-1]++;
				expectedBlockValues[i+1][j+1]++;
				expectedBlockValues[i+1][j-1]++;
				expectedBlockValues[i-1][j+1]++;
			}
			
			#pragma oss task \
				commutative(matrix[i-1][j-1]) \
				commutative(matrix[i+1][j+1]) \
				commutative(matrix[i+1][j-1]) \
				commutative(matrix[i-1][j+1]) \
				label(update edges)
			{
				blockUpdate(i-1, j-1);
				blockUpdate(i+1, j+1);
				blockUpdate(i+1, j-1);
				blockUpdate(i-1, j+1);
			}
		}
	}
}


static void empty_cross_iteration(bool updateExpectedValues)
{
	#pragma oss task weakcommutative(([NUM_BLOCKS] matrix)) label(empty cross iteration)
	for (int i = 1; i < NUM_BLOCKS-1; i++) {
		for (int j = 1; j < NUM_BLOCKS-1; j++) {
			if (updateExpectedValues) {
				expectedBlockValues[i-1][j]++;
				expectedBlockValues[i+1][j]++;
				expectedBlockValues[i][j-1]++;
				expectedBlockValues[i][j+1]++;
			}
			
			#pragma oss task \
				commutative(matrix[i-1][j]) \
				commutative(matrix[i+1][j]) \
				commutative(matrix[i][j-1]) \
				commutative(matrix[i][j+1]) \
				label(update empty cross)
			{
				blockUpdate(i-1, j);
				blockUpdate(i+1, j);
				blockUpdate(i, j-1);
				blockUpdate(i, j+1);
			}
		}
	}
}



static void verifyBlock(int i, int j)
{
	bool good = true;
	
	for (int ii = 0; ii < BLOCK_SIZE; ii++) {
		for (int jj = 0; jj < BLOCK_SIZE; jj++) {
			if (matrix[i][j][ii][jj] != expectedBlockValues[i][j]) {
				good = false;
				break;
			}
		}
	}
	
	std::ostringstream oss;
	oss << "Block [" <<  i << ", " << j << "] has the expected value";
	tap.evaluate(good, oss.str());
}


static void verify()
{
	for (int i = 0; i < NUM_BLOCKS; i++) {
		for (int j = 0; j < NUM_BLOCKS; j++) {
			#pragma oss task in(matrix[i][j]) label(verify block)
			verifyBlock(i, j);
		}
	}
}


int main(int argc, char **argv)
{
	nanos6_wait_for_full_initialization();
	
	long activeCPUs = nanos6_get_num_cpus();
	if (activeCPUs < 2) {
		// This test only works correctly with at least 2 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with less than 2 CPUs");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(NUM_BLOCKS * NUM_BLOCKS * NUM_ROUNDS * 5);
	
	tap.begin();
	
	tap.emitDiagnostic("Test1: _ X _ ");
	tap.emitDiagnostic("Test1: X X X ");
	tap.emitDiagnostic("Test1: _ X _ ");
	for (int round = 0; round < NUM_ROUNDS; round++) {
		init(/* init expected values */ round == 0);
		for (int it = 0; it < NUM_ITERATIONS; it++) {
			cross_iteration(/* update expected values */ round == 0);
		}
		verify();
	}
	#pragma oss taskwait
	
	tap.emitDiagnostic("Test2: _ X _ ");
	tap.emitDiagnostic("Test2: X _ X ");
	tap.emitDiagnostic("Test2: _ _ _ ");
	for (int round = 0; round < NUM_ROUNDS; round++) {
		init(/* init expected values */ round == 0);
		for (int it = 0; it < NUM_ITERATIONS; it++) {
			circumflex_iteration(/* update expected values */ round == 0);
		}
		verify();
	}
	#pragma oss taskwait
	
	tap.emitDiagnostic("Test3: _ _ X ");
	tap.emitDiagnostic("Test3: _ _ _ ");
	tap.emitDiagnostic("Test3: X _ _ ");
	for (int round = 0; round < NUM_ROUNDS; round++) {
		init(/* init expected values */ round == 0);
		for (int it = 0; it < NUM_ITERATIONS; it++) {
			spread_iteration(/* update expected values */ round == 0);
		}
		verify();
	}
	#pragma oss taskwait
	
	tap.emitDiagnostic("Test4: _ X _  |  X _ X");
	tap.emitDiagnostic("Test4: X X X  |  _ _ _");
	tap.emitDiagnostic("Test4: _ X _  |  X _ X");
	for (int round = 0; round < NUM_ROUNDS; round++) {
		init(/* init expected values */ round == 0);
		for (int it = 0; it < NUM_ITERATIONS*2; it++) {
			if ((it % 2) == 0) {
				cross_iteration(/* update expected values */ round == 0);
			} else {
				edges_iteration(/* update expected values */ round == 0);
			}
		}
		verify();
	}
	#pragma oss taskwait
	
	tap.emitDiagnostic("Test5: _ X _ ");
	tap.emitDiagnostic("Test5: X _ X ");
	tap.emitDiagnostic("Test5: _ X _ ");
	for (int round = 0; round < NUM_ROUNDS; round++) {
		init(/* init expected values */ round == 0);
		for (int it = 0; it < NUM_ITERATIONS; it++) {
			empty_cross_iteration(/* update expected values */ round == 0);
		}
		verify();
	}
	#pragma oss taskwait
	
	tap.end();
	
	return 0;
}

