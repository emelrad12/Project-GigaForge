#pragma once
#include "MemoryDroplet.h"
#include "MemoryPool.h"
using namespace GigaEntity;

struct DropletStats
{
	DropletStats(MemoryDroplet droplet)
	{
		freedUpSpace = droplet.freedSpace;
		remainingSpace = droplet.GetRemainingSpace();
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

	int pureDroplets;
	int usedDroplets;
};
