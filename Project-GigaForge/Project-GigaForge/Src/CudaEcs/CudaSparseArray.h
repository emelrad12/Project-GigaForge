#pragma once
#include <vector>

#include "cuda.h"
#include "CudaGlobals.h"
#include "../Ecs/SparseArray.h"

namespace GigaEntity
{
	template <typename T>
	class CudaSparseArray
	{
	public:
		explicit CudaSparseArray(SparseArray<T>& sparseArray) : data(nullptr), chunkCount(sparseArray.chunkCount),
		                                                        chunkSize(sparseArray.chunkSize),
		                                                        totalSize(chunkSize * chunkCount),
		                                                        sparseArray(sparseArray)
		{
			AllocUnManagedArray(data, chunkCount);
			std::vector<T*> tempCpu = std::vector<T*>(chunkCount);
			for (int i = 0; i < chunkCount; i++)
			{
				if (sparseArray.ContainsChunkForItem(i * chunkSize))
				{
					AllocUnManagedArray(tempCpu[i], chunkSize);
					CopyToUnmanagedArray(tempCpu[i], sparseArray.GetChunkFastHandle(i), chunkSize);
				}
			}
			CopyToUnmanagedArray(data, tempCpu.data(), chunkCount);
		}

		void Free()
		{
			std::vector<T*> tempCpu = std::vector<T*>(chunkCount);
			cudaMemcpy(tempCpu.data(), data, sizeof(T*) * chunkCount, cudaMemcpyDeviceToHost);
			for (int i = 0; i < chunkCount; i++)
			{
				if (tempCpu[i] != nullptr)
				{
					cudaFree(tempCpu[i]);
				}
			}
			cudaFree(data);
		}

		void CopyToCpu()
		{
			std::vector<T*> tempCpu = std::vector<T*>(chunkCount);
			cudaMemcpy(tempCpu.data(), data, sizeof(T*) * chunkCount, cudaMemcpyDeviceToHost);
			
			for (int i = 0; i < chunkCount; i++)
			{
				if (sparseArray.ContainsChunkForItem(i * chunkSize))
				{
					CopyToUnmanagedArray(sparseArray.GetChunkFastHandle(i), tempCpu[i], chunkSize);
				}
			}
		}

		__allowDevice__ int GetChunkValidUntil(int chunkId)
		{
			if (data[chunkId] == nullptr)return 0;
			return (chunkId + 1) * chunkSize;
		}

		__allowDevice__ T* GetChunkFastHandle(int chunkId)
		{
			return data[chunkId];
		}

		__allowDevice__ int GetChunkIndex(int itemId) const
		{
			return itemId / chunkSize;
		}

		__allowDevice__ bool ContainsChunkForItem(int itemId)
		{
			itemId /= chunkSize;
			return data[itemId] != nullptr;
		}

		__allowDevice__ void SetAt(T* chunk, int itemIndex, const T& item)
		{
			chunk[itemIndex] = item;
		}

		__allowDevice__ T& operator[](int index)
		{
			auto chunkIndex = index / chunkSize;
			auto itemIndex = index % chunkSize;
			return data[chunkIndex][itemIndex];
		}

		T** data;
		SparseArray<T>& sparseArray;
		const int chunkCount;
		const int chunkSize;
		const int totalSize;
	};
}
