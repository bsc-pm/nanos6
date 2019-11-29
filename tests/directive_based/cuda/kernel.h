#ifdef __cplusplus
extern "C"
{
#endif

#pragma oss task in([n]x) inout([n]y) device(cuda) ndrange(1, n, 128)
__global__ void saxpy(long int n, double a, const double* x, double* y);

#ifdef __cplusplus
}
#endif
