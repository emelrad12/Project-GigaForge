#include <thread>
#include <wrl/module.h>

#include "Timer.h"
#include "cassert"
#include "Src/Ecs/EcsSystem.h"
#include "Src/Ecs/EntityManager.h"
#include "Src/Ecs/CommandBuffer.h"
using namespace GigaEntity;
void TestComp();
template <typename T>
void Print(T data)
{
	std::cout << data << std::endl;
}

void CudaTest();

#define ompLoop omp parallel for schedule(static, 5000 * 100)

struct ArgumentsObject
{
	ArgumentsObject(CommandBuffer buffer, FastHandle(float) floatHandle) : buffer(buffer), floatHandle(floatHandle)
	{
	}

	CommandBuffer& buffer;
	FastHandle(float)& floatHandle;
};

void LambdaFunc(int entityIndex, int& item, ArgumentsObject arguments)
{
	item += entityIndex;
}

auto totalUnsafe = 0;

void ArrTest()
{
	long long totalCount = 1000;
	totalCount *= 1000;
	totalCount *= 1000;
	std::cout << totalCount << std::endl;
	auto armyCount = 10;
	auto unitPerPlayerCount = totalCount / armyCount;
	auto armies = vector<vector<long long>>(armyCount);
	auto armiesResults = vector<int>(armyCount);
	for (auto armyId = 0; armyId < armyCount; armyId++)
	{
		armies[armyId] = vector<long long>(unitPerPlayerCount);
		auto& data = armies[armyId];
		for (auto unitId = 0; unitId < unitPerPlayerCount; unitId++)
		{
			data[unitId] = 1;
		}
	}
	Timer t = Timer();
	auto times = 5;
	for (auto i = 0; i < times; i++)
	{
#pragma omp parallel for
		for (auto armyId = 0; armyId < armyCount; armyId++)
		{
			auto& data = armies[armyId];
			for (auto unitId = 0; unitId < unitPerPlayerCount; unitId++)
			{
				totalUnsafe += data[unitId];
			}
			armiesResults[armyId] = totalUnsafe;
		}
	}

	auto elapsed = t.Stop("test");
	auto gb = 1024 * 1024 * 1024;
	std::cout << totalCount * sizeof(long long) * (1000.0 / elapsed) / gb * times << std::endl;
	for (auto armyId = 0; armyId < armyCount; armyId++)
	{
		std::cout << armiesResults[armyId] << std::endl;
	}
}

int main()
{
	TestComp();
	return 0;
	// ArrTest();
	// return 0;
// 	HINSTANCE hInstance = (HINSTANCE)GetModuleHandle(NULL);
// #if defined(DEBUG) | defined(_DEBUG)
// 	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
// #endif
//
// 	return 0;
	CudaTest();
	Timer timer(false);
	auto manager = EntityManager();
	manager.AddType<int>();
	manager.AddType<float>();
	auto buffer = CommandBuffer();
	buffer.RegisterComponent<int>();
	constexpr auto count = globalCount;
	timer.Start();
	auto task1 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<int>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					
					buffer.AddComponent<int>(index, index, handle);
				}
			}
		}
	};

	std::thread t1(task1);

	t1.join();
	timer.Stop("Tasks");
	timer.Start();
	manager.ExecuteCommands(buffer);
	timer.Stop("Execute");
	auto array1 = manager.GetComponentArray<int>();
	timer.Start();

	auto system = EcsSystem(manager);
	auto newBuffer = CommandBuffer();
	newBuffer.RegisterComponent<float>();
	auto args = ArgumentsObject(newBuffer, newBuffer.GetFastAddHandle<float>());
	auto builder = system.Builder().WithEntities<int>().WithArguments(args);
	auto sys = builder.WithFunction < CastFunction(LambdaFunc) >();
	sys.Run();
	manager.ExecuteCommands(newBuffer);
	timer.Stop("System");

	for (int i = 0; i < count; i++)
	{
		auto val1 = array1[i];
		if (array1[i] != i + i)
		{
			throw "false";
		}
	}
}
