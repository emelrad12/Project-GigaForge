#pragma once
#include "../Ecs/Globals.h"
#include "DropletHandle.h"
#include "MemoryDroplet.h"
#include "queue"
using namespace GigaEntity;
class MemoryPool
{
public:
	MemoryPool()
	{
		auto size = 1000;
		droplets = vector<MemoryDroplet>(size);
		dropletQueue = std::queue<MemoryDroplet*>();
		for (auto i = 0; i < size; i++)
		{
			droplets[i] = MemoryDroplet(i);
			dropletQueue.push(&droplets[i]);
		}
	}

	int chunkSize = 1024;
	vector<MemoryDroplet> droplets;
	std::queue<MemoryDroplet*> dropletQueue;
	std::unordered_map<DropletHandle, DropletMapping> handleMappings;
	std::unordered_map<uint32_t, MemoryDroplet> dropletMappings;
	int nextHandleId;
	
	MemoryDroplet& FindBlock(const int size)
	{
		auto& firstBlock = *dropletQueue.front();
		if (firstBlock.GetRemainingSpace() < size)
		{
			auto& b = dropletQueue.front();
			dropletQueue.pop();
			dropletQueue.emplace(b);
			return FindBlock(size);
		}
		return firstBlock;
	}

	DropletHandle GetHandle()
	{
		return DropletHandle(nextHandleId++);
	}

	void EvaporateDroplet(MemoryDroplet& oldDroplet)
	{
		for (const auto handle : oldDroplet.callerIds)
		{
			auto map = handleMappings[handle];
			auto& newDroplet = FindBlock(map.length);
			auto newMapping = newDroplet.ReserveSpace(map.length, handle);
			handleMappings[handle] = newMapping;
		}
	}
};
