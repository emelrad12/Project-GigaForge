#pragma once
#include "MemoryDroplet.h"
#include "MemoryPool.h"
using namespace GigaEntity;

namespace GigaEntity
{
	struct DropletStats
	{
		DropletStats(MemoryDroplet droplet)
		{
			freedUpSpace = droplet.freedSpace;
			remainingSpace = droplet.GetRemainingSpace();
		}
		
		string GetText()
		{
			return string("Droplet") + "  freedUpSpace:" + std::to_string(freedUpSpace) + "  remainingSpace:" + std::to_string(remainingSpace);
		}
		
		int freedUpSpace;
		int remainingSpace;
	};

	struct MemoryPoolStats
	{
		explicit MemoryPoolStats(MemoryPool pool)
		{
			const auto dropletCount = pool.droplets.size();
			pureDroplets = dropletCount;
			for (size_t i = 0; i < dropletCount; i++)
			{
				if (!pool.droplets[i].IsPure())
				{
					pureDroplets--;
				}
			}
			usedDroplets = dropletCount - pureDroplets;
		}

		string GetText()
		{
			return string("Pool") + "  pure:" + std::to_string(pureDroplets) + "  used:" + std::to_string(usedDroplets);
		}

		int pureDroplets;
		int usedDroplets;
	};
}
