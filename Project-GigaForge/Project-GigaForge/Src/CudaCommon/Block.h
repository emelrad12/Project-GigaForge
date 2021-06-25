#pragma once
#include <unordered_set>
#include "../Ecs/Globals.h"

namespace GigaEntity
{
	class Block
	{
	public:
		char* freeSlotStart;
		char* nextFreeSlot;
		int totalSize = 1024 * 1024;
		int freedSpace = 0;
		std::unordered_set<uint64_t> callerIds;

		Block()
		{
			freeSlotStart = new char[totalSize];
			nextFreeSlot = freeSlotStart;
		}

		void Destroy()
		{
			delete[] freeSlotStart;
		}

		[[nodiscard]] long long GetRemainingSpace() const
		{
			const auto diff = nextFreeSlot - freeSlotStart;
			return totalSize - diff;
		}

		[[nodiscard]] bool ShouldPurge() const
		{
			const auto factor = 1 << 3; //If more than this amount of space is taken then we should probably purge and move the blocks
			return freedSpace > totalSize / factor;
		}

		[[nodiscard]] char* ReserveSpace(const int len, const uint64_t id)
		{
			callerIds.emplace(id);
			nextFreeSlot += len;
			return nextFreeSlot - len;
		}

		void FreeSpace(const int len, const uint64_t id)
		{
			callerIds.erase(id);
			freedSpace += len;
		}
	};
}
