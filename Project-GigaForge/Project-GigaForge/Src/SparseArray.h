#pragma once
#include "../Globals.h"
#include <bitset>

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
