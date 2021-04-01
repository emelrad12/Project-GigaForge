#include "cuda.h"
#include "device_types.h"
#include <cstdio>

#include "CudaEcsSystem.h"
#include "../Ecs/CommandBuffer.h"
#include "../Ecs/EntityManager.h"
#define ompLoop omp parallel for schedule(static, 5000 * 100)

using namespace GigaEntity;
__global__ void cuda_hello()
{
	printf("Hello World from GPU!\n");
}


__device__ void LambdaFunc(int entityIndex, int& item, int arguments)
{
	if (item > entityIndex)
	{
		item--;
	}
	else
	{
		item++;
	}
}

CreateKernelWithFunction(LambdaFunc, Kernel, int, int)

void CudaTest()
{
	auto manager = EntityManager();
	manager.AddType<int>();
	auto buffer = CommandBuffer();
	buffer.RegisterComponent<int>();
	constexpr auto count = DEBUG ? 5000 * 100 : 5000 * 10000;

	auto& handle = buffer.GetFastAddHandle<int>();
	for (int offset = 0; offset < count; offset += chunkSize)
	{
#pragma ompLoop
		for (int i = 0; i < chunkSize; i++)
		{
			const auto index = i + offset;
			if (index < count)
			{
				buffer.AddComponent<int>(index, index, handle);
			}
		}
	}

	manager.ExecuteCommands(buffer);

	auto cudaManager = CudaEntityManager(manager);
	cudaManager.CopyToCuda<int>();
	
	RunKernel(cudaManager, 0);
	cudaDeviceSynchronize();
	checkCudaLastError
}
