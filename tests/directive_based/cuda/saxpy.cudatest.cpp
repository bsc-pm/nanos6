/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#include "../../TestAnyProtocolProducer.hpp"
#include "kernel.h"


#define TOTALSIZE  (1024*1024)
#define BLOCKSIZE  (16)


TestAnyProtocolProducer tap;

#pragma oss task out([BS]x) out([BS]y)
void init_chunk(long int BS, long int start, double *x, double *y){
	for (long int i = 0; i<BS; ++i){
		x[i]= start + i;
		y[i]= start + i + 2;
	}
}

void init_data(long int N, long int BS, double *x, double *y){
	for (long int i = 0; i < N; i+= BS){
		init_chunk(BS, i, &x[i], &y[i]);
	}	
}

void call_saxpy(long int N, long int BS, double a, double *x, double *y){
	for (long int i=0; i<N; i+=BS ){
		saxpy(BS, a, &x[i],  &y[i]);
	}
}

bool check_execution(long int N, double a, double *x, double *y){
	for(long int i = 0; i < N; ++i){
		if(y[i] != a*i+(i+2)){ // There may be doubleing point precision errors in large numbers!
			return false;
		}
	}
	return true;
}

int main() {
 
	tap.registerNewTests(1);
	tap.begin();
	
	// Saxpy parameters
	double a=5, *x, *y;
	int N = TOTALSIZE;
	int tasknum = BLOCKSIZE;
	int BS = N/tasknum;

	// Allocate regular memory
	x = (double *) malloc(N*sizeof(double));
	y = (double *) malloc(N*sizeof(double));
	
	init_data(N, BS, x, y);	
	call_saxpy(N, BS, a, x, y);
	#pragma oss taskwait
		

	check_execution(N, a, x, y);

	bool validates = check_execution(N, a, x, y);
;
	
	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	tap.end();
	free(x);
	free(y);
	return 0;
}