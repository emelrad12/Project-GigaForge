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

		void Set(int index, T item)
		{
			if (!data.ContainsChunkForItem(index))
			{
				data.AllocateChunkForItem(index);
				entityContains.AllocateChunkForItem(index);
			}
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
