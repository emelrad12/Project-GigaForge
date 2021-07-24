#pragma once
#include "../Src/CudaCommon/FastMemoryManager/MemoryLake.h"
#include "../Src/CudaCommon/FastMemoryManager/MemStats.h"
using namespace GigaEntity;

inline void TestMemory()
{
	auto mem = MemoryLake();
	auto queue = std::vector<DropletHandle>();
	for (size_t i = 0; i < 512 * 32; i++)
	{
		auto handle = mem.ReserveSpace(1024 * 8);
		queue.push_back(handle);
		auto ptr = mem.GetPointerFromHandle(handle);
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
