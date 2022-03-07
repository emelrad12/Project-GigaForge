#include "device_types.h"
#include "doctest.h"
#include "../src/CudaEcs/CudaGlobals.h"

#if defined (__INTELLISENSE__) | defined (__RESHARPER__)
template <class T1, class T2>
__device__ int atomicAdd(T1 x, T2 y);
#endif

__device__ uint32_t hash(uint32_t a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


__global__ void SomeCudaFunc(int** buckets, int* indices, int bucketSize, int len)
{
	const auto index = getCudaIndex;
	const auto newBucketIndex = hash(index) % len;
	const auto newIndexInBucket = atomicAdd(indices + newBucketIndex, 1);
#ifndef NDEBUG
	printf("(%d  %d  %d) \n", newBucketIndex, newIndexInBucket, index);
#endif
	if (newIndexInBucket >= bucketSize) return;
	const auto bucket = buckets[newBucketIndex];
	bucket[newIndexInBucket] = newIndexInBucket;
}

TEST_CASE("Test Other Cuda")
{
#ifndef NDEBUG
	auto len = 10;
	auto bucketSize = 10;
#else
	auto len = 1000 * 100;
	auto bucketSize = 1000;
#endif

	auto buckets = new int*[len];
	int* bigArray;
	auto bigArrayLen = bucketSize * len;
	AllocUnManagedArray(bigArray, bigArrayLen);
	for (size_t i = 0; i < len; i++)
	{
		buckets[i] = bigArray + bucketSize * i;
	}
	int** cudaBucketPtr;
	int* cudaBucketIndices;
	AllocUnManagedArray(cudaBucketPtr, len);
	AllocUnManagedArray(cudaBucketIndices, len);
	CopyToUnmanagedArray(cudaBucketPtr, buckets, len);
	auto threads = 256;
	auto blocks = (bigArrayLen + threads - 1) / threads;
	SomeCudaFunc << <blocks, threads >> >(cudaBucketPtr, cudaBucketIndices, bucketSize, len);
	checkCudaError(cudaDeviceSynchronize())
	const auto resultArray = new int[bigArrayLen];
	memset(resultArray, -500, bigArrayLen);
	CopyToUnmanagedArray(resultArray, bigArray, bigArrayLen);
	checkCudaError(cudaDeviceSynchronize())
	auto counter = 0;
	for (size_t i = 0; i < bigArrayLen; i++)
	{
		auto value = resultArray[i];
		if (value != 0)
		{
			counter++;
		}
#ifndef NDEBUG
		std::cout << value << "\n";
#endif
	}
	std::cout << counter << "\n";
}
