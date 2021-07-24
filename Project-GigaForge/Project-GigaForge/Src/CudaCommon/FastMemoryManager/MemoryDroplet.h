#pragma once
#include <unordered_set>
#include "DropletHandle.h"
#include "../../CudaEcs/CudaGlobals.h"

namespace GigaEntity
{
	class MemoryDroplet
	{
	public:
		char* memoryLocation;
		uint32_t totalSize = 1024 * 1024;
		uint16_t id;
		int freedSpace = 0;
		uint32_t nextFreeSlot;
		std::unordered_set<DropletHandle> callerIds;
		bool onDevice;

		MemoryDroplet()
		{
		}

		explicit MemoryDroplet(uint32_t id, bool allocateOnDevice) : id(id)
		{
			onDevice = allocateOnDevice;
			if (allocateOnDevice)
			{
				AllocUnManagedArray(memoryLocation, totalSize);
			}
			else
			{
				memoryLocation = new char[totalSize];
			}
			nextFreeSlot = 0;
		}

		bool IsPure() const
		{
			return nextFreeSlot == 0;
		}

		void Clear()
		{
			nextFreeSlot = 0;
			callerIds.clear();
			freedSpace = 0;
		}

		[[nodiscard]] uint32_t GetRemainingSpace() const
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
