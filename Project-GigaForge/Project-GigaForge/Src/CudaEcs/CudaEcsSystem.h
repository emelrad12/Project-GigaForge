#pragma once
#include "CudaComponentArray.h"
#include "CudaEntityManager.h"
#include "../Ecs/EntityManager.h"
#include "CudaGlobals.h"
#include <nvfunctional>

using std::tuple;
__device__ void VoidFunc()
{
};

#define CastFunction(func) reinterpret_cast<void(*)()>(func)

#define CreateKernelWithFunction(func, name, TArgumentObject, E1) \
	__global__ void name(CudaComponentArray<E1> arr1, int size, TArgumentObject argumentObject){ \
	auto index = getCudaIndex;\
	if(index >= size)return;\
	if(arr1.ContainsEntity(index))\
	func(getCudaIndex, arr1[getCudaIndex], argumentObject); \
}\
void Run##name(CudaEntityManager manager, TArgumentObject argumentObject)\
{\
	auto arr1 = manager.GetComponentArray<E1>();\
	auto threads = 256;\
	auto blocks = (manager.itemCount + threads - 1) / threads;\
	name << < blocks, threads >> > (arr1, manager.itemCount, argumentObject);\
	checkCudaLastError;\
}