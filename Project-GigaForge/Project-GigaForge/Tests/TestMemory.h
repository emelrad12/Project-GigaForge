#pragma once
#include "../Src/CudaCommon/MemoryPool.h"
#include "../Src/CudaCommon/MemStats.h"
using namespace GigaEntity;

inline void TestMemory()
{
	auto mem = MemoryPool();
	auto queue = std::vector<DropletHandle>();
	for (size_t i = 0; i < 512 * 32; i++)
	{
		queue.push_back(mem.ReserveSpace(1024 * 8));
	}
	for (auto item : queue)
	{
		mem.ReleaseHandle(item);
	}

	auto memStats = MemoryPoolStats(mem);
	std::cout << memStats.GetText() << std::endl;
	for (auto& droplet : mem.droplets)
	{
		if (!droplet.IsPure())
		{
			auto dropletStats = DropletStats(droplet);
			std::cout << dropletStats.GetText() << std::endl;
		}
	}
	mem.ConsolidateDroplets();
	memStats = MemoryPoolStats(mem);
	std::cout << memStats.GetText() << std::endl;

	for (auto& droplet : mem.droplets)
	{
		if (!droplet.IsPure())
		{
			auto dropletStats = DropletStats(droplet);
			std::cout << dropletStats.GetText() << std::endl;
		}
	}
}
