#pragma once
#include "../src/CudaCommon/MemoryManager.h"
inline void TestMemory()
{
	auto mem = MemoryManager();
	for (size_t i = 0; i < 1024; i++)
	{
		auto& block = mem.FindBlock(1024);
		auto ptr= block.ReserveSpace(1024, i);
		i++;
		i--;
	}
}