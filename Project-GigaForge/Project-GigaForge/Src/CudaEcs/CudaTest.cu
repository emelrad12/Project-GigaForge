#include "cuda.h"
#include "device_types.h"
#include <cstdio>

__global__ void cuda_hello()
{
	printf("Hello World from GPU!\n");
}

void CudaTest()
{
	cuda_hello << <1, 1 >> >();
}
