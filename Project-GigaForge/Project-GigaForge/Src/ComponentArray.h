#pragma once
#include "SparseArray.h"

template <typename T>
class ComponentArray
{
public:
	ComponentArray(int chunkCount, int chunkSize) : chunkCount(chunkCount),
	                                                chunkSize(chunkSize),
	                                                totalSize(chunkSize * chunkCount),
	                                                data(chunkCount, chunkSize),
	                                                entityContains(chunkCount, chunkSize)
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

	SparseArray<T> data;
	SparseArray<bool> entityContains;

	const int chunkCount;
	const int chunkSize;
	const int totalSize;
};
