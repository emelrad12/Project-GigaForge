#pragma once
#include "cuda.h"
#include "CudaGlobals.h"
#include "../Ecs/SparseArray.h"

namespace GigaEntity
{
	template <typename T>
	class CudaSparseArray
	{
	public:
		CudaSparseArray(SparseArray<T>& sparseArray) : chunkCount(sparseArray.chunkCount),
		                                               chunkSize(sparseArray.chunkSize),
		                                               totalSize(chunkSize * chunkCount)
		{
			AllocUnManagedArray(data, chunkCount);
			auto tempData = new T*[chunkCount];
			for (size_t i = 0; i < chunkCount; i++)
			{
				AllocUnManagedArray(tempData[i], chunkSize);
			}
			CopyToUnmanagedArray(data, tempData, chunkCount);
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
		const int chunkCount;
		const int chunkSize;
		const int totalSize;
	};
}
