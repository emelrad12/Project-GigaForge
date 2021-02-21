#pragma once
#include "SparseArray.h"

namespace GigaEntity
{
	template <typename T>
	struct CombinedChunk
	{
		CombinedChunk(int end, int chunkStart, T* data, bool* entityContains) : end(end), start(chunkStart),
			data(data), entityContains(entityContains)
		{
		}

		int end;
		int start;
		T* data;
		bool* entityContains;

		static CombinedChunk<T> GetDefault()
		{
			return CombinedChunk<T>(-1, 0, nullptr, nullptr);
		}
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

		void SetAt(CombinedChunk<T> combinedChunk, int itemIndex, T& item)
		{
			auto newIndex = itemIndex - combinedChunk.start;
			data.SetAt(combinedChunk.data, newIndex, item);
			entityContains.SetAt(combinedChunk.entityContains, newIndex, true);
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
