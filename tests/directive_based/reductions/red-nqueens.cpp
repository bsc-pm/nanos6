/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test N-Queens benchmark without taskwaits in nested reductions
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include "TestAnyProtocolProducer.hpp"


#define FINAL_DEPTH 9
#define N 9

// #define EXPECTED_RESULT 10       // 5
// #define EXPECTED_RESULT 4        // 6
// #define EXPECTED_RESULT 40       // 7
// #define EXPECTED_RESULT 92       // 8
#define EXPECTED_RESULT 352      // 9
// #define EXPECTED_RESULT 724      // 10
// #define EXPECTED_RESULT 2680     // 11
// #define EXPECTED_RESULT 14200    // 12
// #define EXPECTED_RESULT 73712    // 13
// #define EXPECTED_RESULT 365596   // 14
// #define EXPECTED_RESULT 2279184  // 15
// #define EXPECTED_RESULT 14772512 // 16


typedef struct sol_node
{
	int row;
	struct sol_node *prev;
} sol_node_t, *sol_t;

// Check if possition can be attacked (1) by currently positioned queens.
// The check is performed column-wise from right to left, checking the latest
// positioned queen, then the previous, etc.
static inline int check_attack(const int col, const int row, sol_t sol)
{
	int j;
	for (j = 0; j < col; j++)
	{
		const int tmp = abs(sol->row - row);
		if (tmp == 0 || tmp == j + 1)
			return 1;
		
		sol = sol->prev;
	}
	return  0;
}

int final_depth = FINAL_DEPTH;
void solve(int n, const int col, sol_node_t& sol, int& result)
{
	if (col == n)
	{
		#pragma oss task reduction(+: result)
		{
			result++;
		}
	}
	else
	{
		for (int row = 0; row < n; row++)
		{
			if (!check_attack(col, row, &sol))
			{
				sol_node_t new_sol;
				new_sol.prev = &sol;
				new_sol.row = row;
				
				#pragma oss task final(final_depth <= col) weakreduction(+: result) label(rec_solve)
				{
					solve(n, col + 1, new_sol, result);
				}
			}
		}
	}
}

TestAnyProtocolProducer tap;
	
int main()
{
	int n = N;
	
	assert(n > 0);
	
	tap.registerNewTests(1);
	tap.begin();
	
	sol_node_t initial_node = {-1, 0};
	int count_main = 0;
	struct timeval start;
	
	gettimeofday(&start, NULL);
	#pragma oss task weakreduction(+:count_main) label(solve)
	{
		solve(n, 0, initial_node, count_main);
	}
	
	#pragma oss task in(count_main) label(print)
	{
		struct timeval stop;
		gettimeofday(&stop, NULL);
		unsigned long elapsed = 1000000*(stop.tv_sec - start.tv_sec);
		elapsed += stop.tv_usec - start.tv_usec;
		
		std::ostringstream oss;
		oss << "Expected result: size = " << n << ", final_depth = " <<
			final_depth << ", time (ms) = " << elapsed/1000 << ", result = " <<
			count_main;
			
		tap.evaluate(count_main == EXPECTED_RESULT, oss.str());
	}
	
	#pragma oss taskwait
	
	tap.end();
	
	return 0;
}
