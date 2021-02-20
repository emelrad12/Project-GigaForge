#pragma once
#include "SparseArray.h"

namespace GigaEntity
{
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

		int GetValidChunkUntil(int index)
		{
			return data.GetChunkValidUntil(index);
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
