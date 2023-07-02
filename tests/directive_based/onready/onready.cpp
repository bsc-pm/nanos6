/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2023 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>
#include <nanos6/debug.h>

#include <unistd.h>

#include "TestAnyProtocolProducer.hpp"

TestAnyProtocolProducer tap;

void onreadyFunction1(int *A, int *B)
{
	tap.evaluate(*A == 11, "Check variable A (11)");
	tap.evaluate(*B == 20, "Check variable B (20)");
	*A += 10;
	*B += 10;
}

void onreadyFunction2(int *A, int *B)
{
	tap.evaluate(*A == 42, "Check variable A (42)");
	tap.evaluate(*B == 40, "Check variable B (40)");
	*A += 10;
	*B += 10;
}

int main()
{
	tap.registerNewTests(14);
	tap.begin();

	int A = 10;
	int B = 20;

	#pragma oss task inout(A) shared(tap) onready(tap.evaluate(A == 10, "Check variable A (10)"))
	{
		sleep(1);
		A++;
	}

	#pragma oss task inout(A) firstprivate(B) onready(onreadyFunction1(&A, &B))
	{
		tap.evaluate(A == 21, "Check variable A (21)");
		tap.evaluate(B == 30, "Check variable B (30)");
	}

	#pragma oss taskwait

	tap.evaluate(A == 21, "Check variable A (21)");
	tap.evaluate(B == 20, "Check variable B (20)");

	A += 20;
	B += 20;

	#pragma oss task inout(A) shared(tap) onready(tap.evaluate(A == 41, "Check variable A (41)"))
	{
		sleep(1);
		A++;
	}

	#pragma oss task inout(A) firstprivate(B) if(0) onready(onreadyFunction2(&A, &B))
	{
		tap.evaluate(A == 52, "Check variable A (52)");
		tap.evaluate(B == 50, "Check variable B (50)");
	}

	#pragma oss taskwait

	tap.evaluate(A == 52, "Check variable A (52)");
	tap.evaluate(B == 40, "Check variable B (40)");

	tap.end();

	return 0;
}
