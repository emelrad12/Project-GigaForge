#pragma once
#include "CudaSparseArray.h"
#include "../Ecs/ComponentArray.h"

namespace GigaEntity
{
	template <typename T>
	struct CudaCombinedChunk
	{
		CudaCombinedChunk(int end, int chunkStart, T* data, bool* entityContains) : start(chunkStart), end(end),
		                                                                            data(data), entityContains(entityContains)
		{
		}

		int start;
		int end;
		T* data;
		bool* entityContains;

		static CudaCombinedChunk<T> GetDefault()
		{
			return CudaCombinedChunk<T>(-1, 0, nullptr, nullptr);
		}
	};

	template <typename T>
	class CudaComponentArray
	{
	public:
		CudaComponentArray(ComponentArray<T> dataToCopyFrom) : data(dataToCopyFrom.data),
		                                                       entityContains(dataToCopyFrom.entityContains),
		                                                       chunkCount(dataToCopyFrom.chunkCount),
		                                                       chunkSize(dataToCopyFrom.chunkSize),
		                                                       totalSize(chunkSize * chunkCount)
		{
		}

		__allowDevice__ bool ContainsEntity(int entity)
		{
			if (entityContains.ContainsChunkForItem(entity))
			{
				return entityContains[entity]; //todo optimize
			}
			return false;
		}

		__allowDevice__ T& operator[](int index)
		{
			return data[index];
		}

		__allowDevice__ CudaCombinedChunk<T> GetCombinedChunk(int index)
		{
			auto chunkId = data.GetChunkIndex(index);
			auto returnValue = CombinedChunk<T>(data.GetChunkValidUntil(chunkId),
			                                    chunkId * chunkSize,
			                                    data.GetChunkFastHandle(chunkId),
			                                    entityContains.GetChunkFastHandle(chunkId));
			if (index - returnValue.start < 0)
			{
				__debugbreak();
			}
			return returnValue;
		}

		__allowDevice__ void SetAt(CudaCombinedChunk<T> combinedChunk, int itemIndex, T& item)
		{
			auto newIndex = itemIndex - combinedChunk.start;
			data.SetAt(combinedChunk.data, newIndex, item);
			entityContains.SetAt(combinedChunk.entityContains, newIndex, true);
		}

		__allowDevice__ void Set(int index, T item)
		{
			data[index] = item;
			entityContains[index] = true;
		}

		CudaSparseArray<T> data;
		CudaSparseArray<bool> entityContains;

		const int chunkCount;
		const int chunkSize;
		const int totalSize;
	};
}
