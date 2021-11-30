#pragma once
#include "doctest.h"
#include "../Src/Ecs/EcsSystem.h"
#include "../Src/Ecs/EntityManager.h"
#include <thread>
#define ompLoop omp parallel for schedule(static, 5000 * 100)

TEST_CASE("TestEcs")
{
	// 	Timer timer(false);
// 	auto manager = EntityManager();
// 	manager.AddType<int>();
// 	manager.AddType<float>();
// 	auto buffer = CommandBuffer();
// 	buffer.RegisterComponent<int>();
// 	constexpr auto count = globalCount;
// 	timer.Start();
// 	auto task1 = [&buffer]()
// 	{
// 		auto& handle = buffer.GetFastAddHandle<int>();
// 		for (int offset = 0; offset < count; offset += chunkSize)
// 		{
// #pragma ompLoop
// 			for (int i = 0; i < chunkSize; i++)
// 			{
// 				const auto index = i + offset;
// 				if (index < count)
// 				{
// 					
// 					buffer.AddComponent<int>(index, index, handle);
// 				}
// 			}
// 		}
// 	};
//
// 	std::thread t1(task1);
//
// 	t1.join();
// 	timer.Stop("Tasks");
// 	timer.Start();
// 	manager.ExecuteCommands(buffer);
// 	timer.Stop("Execute");
// 	auto array1 = manager.GetComponentArray<int>();
// 	timer.Start();
//
// 	auto system = EcsSystem(manager);
// 	auto newBuffer = CommandBuffer();
// 	newBuffer.RegisterComponent<float>();
// 	auto args = ArgumentsObject(newBuffer, newBuffer.GetFastAddHandle<float>());
// 	auto builder = system.Builder().WithEntities<int>().WithArguments(args);
// 	auto sys = builder.WithFunction < CastFunction(LambdaFunc) >();
// 	sys.Run();
// 	manager.ExecuteCommands(newBuffer);
// 	timer.Stop("System");
//
// 	for (int i = 0; i < count; i++)
// 	{
// 		auto val1 = array1[i];
// 		if (array1[i] != i + i)
// 		{
// 			throw "false";
// 		}
// 	}
}