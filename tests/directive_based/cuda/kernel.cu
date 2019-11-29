#include "kernel.h"

#include <stdio.h>

__global__ void saxpy(long int n, double a, const double* x, double* y)
{
   long int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < n) y[i] = a * x[i] + y[i];
}
