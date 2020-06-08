/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>

#include "TestAnyProtocolProducer.hpp"

#define TOTAL_SIZE 9
#define FINAL_DEPTH 4
#define GRAINSIZE 2
#define CORRECT_VALUE 352

typedef struct sol_node {
	int row;
	struct sol_node *prev;
} sol_node_t, *sol_t;


TestAnyProtocolProducer tap;

int count;
int final_depth;
int grainsize;


// Check if possition can be attacked (1) by currently positioned queens.
// The check is performed column-wise from right to left, checking the latest
// positioned queen, then the previous, etc.
static inline int check_attack(const int col, const int row, sol_t sol)
{
	for (int j = 0; j < col; j++) {
		const int tmp = abs(sol->row - row);
		if (tmp == 0 || tmp == j + 1) {
			return 1;
		}
		sol = sol->prev;
	}
	return  0;
}

void solve(int n, const int col, sol_node_t& sol)
{
	if (col == n) {
		__sync_fetch_and_add(&count, 1);
	} else {
		#pragma oss taskloop for grainsize(grainsize)
		for (int row = 0; row < n; row++) {
			if (!check_attack(col, row, &sol)) {
				sol_node_t new_sol;
				new_sol.prev = &sol;
				new_sol.row = row;

				solve(n, col + 1, new_sol);
			}
		}
		#pragma oss taskwait
	}
}

int main() {
	int n = TOTAL_SIZE;
	final_depth = FINAL_DEPTH;
	grainsize = GRAINSIZE;
	count = 0;

	assert(n > 0);
	sol_node_t initial_node = {-1, 0};

	tap.registerNewTests(1);
	tap.begin();

	#pragma oss task
	solve(n, 0, initial_node);
	#pragma oss taskwait

	tap.evaluate(count == CORRECT_VALUE, "The result of the nqueens program is correct");

	tap.end();

	return 0;
}
