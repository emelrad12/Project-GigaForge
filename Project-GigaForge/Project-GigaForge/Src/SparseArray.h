#pragma once
#include "../Globals.h"
#include <bitset>

namespace GigaEntity
{
	template <typename T>
	class SparseArray
	{
	public:
		SparseArray(int chunkCount, int chunkSize) : chunkCount(chunkCount),
		                                             chunkSize(chunkSize),
		                                             totalSize(chunkSize * chunkCount)
		{
			data = new T*[chunkCount];
			for (size_t i = 0; i < chunkCount; i++)
			{
				data[i] = nullptr;
			}
		}

		void FreeChunk(int chunkId)
		{
			delete[] data[chunkId];
		}

		int GetChunkValidUntil(int chunkId)
		{
			if (data[chunkId] == nullptr)return 0;
			return (chunkId + 1) * chunkSize;
		}

		T* GetChunkFastHandle(int chunkId)
		{
			return data[chunkId];
		}
		
		int GetChunkIndex(int itemId) const
		{
			return itemId / chunkSize;
		}

		bool ContainsChunkForItem(int itemId)
		{
			itemId /= chunkSize;
			return data[itemId] != nullptr;
		}

		void AllocateChunkForItem(int itemId)
		{
			itemId /= chunkSize;
			data[itemId] = new T[chunkSize];
		}

		void SetAt(T* chunk, int itemIndex, const T& item)
		{
			chunk[itemIndex] = item;
		}

		T& operator[](int index)
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
