#pragma once
#include "../Src/CudaCommon/MemoryPool.h"

inline void TestMemory()
{
	auto mem = MemoryPool();
	for (size_t i = 0; i < 512; i++)
	{
		auto handle = mem.GetHandle();
		auto& block = mem.FindBlock(1024);
		auto ptr = block.ReserveSpace(1024, handle);
		i++;
		i--;
	}
	auto& b = mem.FindBlock(1024);
	mem.EvaporateDroplet(b);
}
