#pragma once
#include <unordered_set>
#include "../Ecs/Globals.h"
#include "DropletHandle.h"

namespace GigaEntity
{
	class MemoryDroplet
	{
	public:
		char* memoryLocation;
		uint32_t nextFreeSlot;
		int totalSize = 1024 * 1024;
		int freedSpace = 0;
		uint16_t id;
		std::unordered_set<DropletHandle> callerIds;

		MemoryDroplet()
		{
		
		}
		
		MemoryDroplet(uint32_t id) : id(id)
		{
			memoryLocation = new char[totalSize];
			nextFreeSlot = 0;
		}

		void Destroy()
		{
			delete[] memoryLocation;
		}

		[[nodiscard]] long long GetRemainingSpace() const
		{
			return totalSize - nextFreeSlot;
		}

		[[nodiscard]] bool ShouldPurge() const
		{
			const auto factor = 1 << 3; //If more than this amount of space is taken then we should probably purge and move the dropletQueue
			return freedSpace > totalSize / factor;
		}

		[[nodiscard]] DropletMapping ReserveSpace(const uint16_t len, const DropletHandle newHandle)
		{
			callerIds.emplace(newHandle);
			auto newMapping = DropletMapping();
			newMapping.length = len;
			newMapping.dropletId = id;
			newMapping.offset = nextFreeSlot;
			nextFreeSlot += len;
			return newMapping;
		}

		void FreeSpace(const int len, const DropletHandle handle)
		{
			callerIds.erase(handle);
			freedSpace += len;
		}
	};
}
