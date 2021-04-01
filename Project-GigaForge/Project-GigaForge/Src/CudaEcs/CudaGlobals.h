#pragma once
#include "cuda.h"
#include "device_types.h"
#include "cuda_runtime.h"
#include "iostream"
#include "device_launch_parameters.h"

#define __allowDevice__ __device__

#define getCudaIndex (threadIdx.x + blockDim.x * blockIdx.x)
#define printCudaError \
	{if (cudaSuccess != error)\
	{\
		std::cout << "Error!" << std::endl;\
		std::cout << cudaGetErrorString(error) << std::endl;\
		__debugbreak();\
	}}

#define checkCudaLastError\
	{const auto error = cudaGetLastError();\
		printCudaError}

#define checkCudaError(item)\
	{const auto error = item;\
		printCudaError}

template <typename T>
void AllocUnManagedArray(T*& item, int size)
{
	checkCudaError(cudaMalloc(&item, size * sizeof(*item)));
}

template <typename T>
void CopyToUnmanagedArray(T* to, T* from, int size)
{
	checkCudaError(cudaMemcpy(to, from, size * sizeof(T), cudaMemcpyDefault));
}

template <typename T>
void CopyToUnmanagedArraySingle(T* to, T* from, int offset)
{
	checkCudaError(cudaMemcpy(to + offset, from + offset, 1 * sizeof(T), cudaMemcpyDefault));
}
