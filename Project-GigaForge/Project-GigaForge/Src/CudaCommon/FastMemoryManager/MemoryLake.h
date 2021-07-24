#pragma once
#include "DropletHandle.h"
#include "MemoryDroplet.h"
#include "queue"
using namespace GigaEntity;

namespace GigaEntity
{
	class MemoryLake
	{
	public:
		MemoryLake()
		{
			auto size = 1000;
			droplets = vector<MemoryDroplet>(size);
			dropletQueue = std::queue<MemoryDroplet*>();
			for (auto i = 0; i < size; i++)
			{
				droplets[i] = MemoryDroplet(i,true);
				dropletQueue.push(&droplets[i]);
			}
		}

		int chunkSize = 1024;
		vector<MemoryDroplet> droplets;
		std::queue<MemoryDroplet*> dropletQueue;
		std::unordered_map<DropletHandle, DropletMapping> handleMappings;
		int nextHandleId;

		DropletHandle ReserveSpace(const int size)
		{
			auto handle = GetHandle();
			auto& block = FindBlock(size);
			auto mapping = block.ReserveSpace(size, handle);
			handleMappings[handle] = mapping;
			return handle;
		}

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

		void ReleaseHandle(DropletHandle handle)
		{
			auto& mapping = GetMappingFromHandle(handle);
			auto& droplet = droplets[mapping.dropletId];
			droplet.FreeSpace(mapping.length, handle);
		}

		char* GetPointerFromHandle(DropletHandle handle)
		{
			return GetPointerFromMapping(GetMappingFromHandle(handle));
		}
		
		char* GetPointerFromMapping(DropletMapping mapping)
		{
			auto droplet = droplets[mapping.dropletId];
			return droplet.memoryLocation + mapping.offset;
		}

		DropletMapping& GetMappingFromHandle(DropletHandle handle)
		{
			return handleMappings[handle];
		}

		DropletHandle GetHandle()
		{
			return DropletHandle(nextHandleId++);
		}

		void ConsolidateDroplets()
		{
			for (auto& droplet : droplets)
			{
				if (!droplet.IsPure() && droplet.ShouldPurge())
				{
					EvaporateDroplet(droplet);
				}
			}
		}

		void EvaporateDroplet(MemoryDroplet& oldDroplet)
		{
			for (const auto handle : oldDroplet.callerIds)
			{
				auto oldMapping = handleMappings[handle];
				auto& newDroplet = FindBlock(oldMapping.length);
				const auto newMapping = newDroplet.ReserveSpace(oldMapping.length, handle);
				handleMappings[handle] = newMapping;
				memcpy(GetPointerFromMapping(newMapping), GetPointerFromMapping(oldMapping), oldMapping.length);
			}

			oldDroplet.Clear();
		}
	};
}
