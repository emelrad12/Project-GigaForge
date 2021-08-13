#pragma once
#include "CudaComponentArray.h"
#include "CudaEntityManager.h"
#include "../Ecs/EntityManager.h"
#include "CudaGlobals.h"
#include <nvfunctional>

using std::tuple;
__device__ inline void VoidFunc()
{
};

#define CastFunction(func) reinterpret_cast<void(*)()>(func)

template <typename E1, typename TArgumentObject, void(*func)(int, E1&, TArgumentObject)>
__global__ void Kernel1(GigaEntity::CudaComponentArray<E1> arr1, int size, TArgumentObject argumentObject)
{
	auto index = getCudaIndex;
	if (index >= size)return;
	if (arr1.ContainsEntity(index))
		func(index, arr1[index], argumentObject);
}

#define CreateKernelWithFunction(func, name, TArgumentObject, E1) \
void Run##name(CudaEntityManager manager, TArgumentObject argumentObject)\
{\
	auto arr1 = manager.GetComponentArray<E1>();\
	auto threads = 256;\
	auto blocks = (manager.itemCount + threads - 1) / threads;\
	Kernel1<E1, TArgumentObject, func> << < blocks, threads >> > (arr1, manager.itemCount, argumentObject);\
	checkCudaLastError;\
}


// #define CreateKernelWithFunction(func, name, TArgumentObject, E1) \
// __global__ void name(CudaComponentArray<E1> arr1, int size, TArgumentObject argumentObject){ \
// 	auto index = getCudaIndex;\
// 	if(index >= size)return;\
// 	if(arr1.ContainsEntity(index))\
// 	func(index, arr1[index], argumentObject); \
// }\
// void Run##name(CudaEntityManager manager, TArgumentObject argumentObject)\
// {\
// 	auto arr1 = manager.GetComponentArray<E1>();\
// 	auto threads = 256;\
// 	auto blocks = (manager.itemCount + threads - 1) / threads;\
// 	name << < blocks, threads >> > (arr1, manager.itemCount, argumentObject);\
// 	checkCudaLastError;\
// }
