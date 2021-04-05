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
		                                               totalSize(chunkSize * chunkCount),
		                                               sparseArray(sparseArray)
		{
			AllocUnManagedArray(data, chunkCount);
			dataCpu = new T*[chunkCount];
			for (int i = 0; i < chunkCount; i++)
			{
				if (sparseArray.ContainsChunkForItem(i * chunkSize))
				{
					AllocUnManagedArray(dataCpu[i], chunkSize);
					CopyToUnmanagedArray(dataCpu[i], sparseArray.GetChunkFastHandle(i), chunkSize);
				}
			}
			CopyToUnmanagedArray(data, dataCpu, chunkCount);
		}

		void Free()
		{
			cudaFree(data);
			for (int i = 0; i < chunkCount; i++)
			{
				if (dataCpu[i] != nullptr)
				{
					cudaFree(dataCpu[i]);
				}
			}
		}

		void CopyToCpu()
		{
			for (int i = 0; i < chunkCount; i++)
			{
				if (sparseArray.ContainsChunkForItem(i * chunkSize))
				{
					CopyToUnmanagedArray(sparseArray.GetChunkFastHandle(i), dataCpu[i], chunkSize);
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

		T** dataCpu;
		T** data;
		SparseArray<T> sparseArray;
		const int chunkCount;
		const int chunkSize;
		const int totalSize;
	};
}
