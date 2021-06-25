#pragma once
#include "../Ecs/Globals.h"
#include <algorithm>
#include "Block.h"
#include "queue"
class MemoryManager
{
public:
	MemoryManager()
	{
		auto size = 1000;
		blocks = std::queue<GigaEntity::Block>();
		for (size_t i = 0; i < size; i++)
		{
			blocks.emplace();
		}
	}
	
	int chunkSize = 1024;
	std::queue<GigaEntity::Block> blocks;

	GigaEntity::Block& FindBlock(const int size)
	{
		auto& firstBlock = blocks.front();
		if(firstBlock.GetRemainingSpace() < size)
		{
			auto b = blocks.front();
			blocks.pop();
			blocks.emplace(b);
			return FindBlock(size);
		}
		return firstBlock;
	}
};
