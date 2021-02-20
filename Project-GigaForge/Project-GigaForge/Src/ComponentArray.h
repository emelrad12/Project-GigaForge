#pragma once
#include "SparseArray.h"

namespace GigaEntity
{
	template <typename T>
	struct CombinedChunk
	{
		CombinedChunk(int end, int chunkStart, T* data, bool* entityContains) : end(end), chunkStart(chunkStart),
			data(data), entityContains(entityContains)
		{
		}

		int end;
		int chunkStart;
		T* data;
		bool* entityContains;
	};

	template <typename T>
	class ComponentArray
	{
	public:
		ComponentArray(int chunkCount, int chunkSize) : data(chunkCount, chunkSize),
		                                                entityContains(chunkCount, chunkSize),
		                                                chunkCount(chunkCount),
		                                                chunkSize(chunkSize),
		                                                totalSize(chunkSize * chunkCount)
		{
		}

		bool ContainsEntity(int entity)
		{
			return entityContains[entity];
		}

		T& operator[](int index)
		{
			return data[index];
		}

		CombinedChunk<T> GetCombinedChunk(int index)
		{
			auto chunkId = data.GetChunkIndex(index);
			return CombinedChunk<T>(data.GetChunkValidUntil(chunkId),
			                        chunkId * chunkSize,
			                        data.GetChunkFastHandle(chunkId),
			                        entityContains.GetChunkFastHandle(chunkId));
		}

		void SetAt(CombinedChunk<T> combinedChunk, int itemIndex, T& item)
		{
			itemIndex -= combinedChunk.chunkStart;
			data.SetAt(combinedChunk.data, itemIndex, item);
			entityContains.SetAt(combinedChunk.entityContains, itemIndex, true);
		}

		void AllocateChunk(int index)
		{
			data.AllocateChunkForItem(index);
			entityContains.AllocateChunkForItem(index);
		}

		void Set(int index, T item)
		{
			data[index] = item;
			entityContains[index] = true;
		}

		SparseArray<T> data;
		SparseArray<bool> entityContains;

		const int chunkCount;
		const int chunkSize;
		const int totalSize;
	};
}
