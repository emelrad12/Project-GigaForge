#include <thread>
#include <wrl/module.h>

#include "Timer.h"
#include "cassert"
#include "Src/Ecs/EcsSystem.h"
#include "Src/Ecs/EntityManager.h"
#include "Src/Ecs/CommandBuffer.h"
#include "Src/Rendering/Init3DApp.h"
#include "Src/Rendering/RenderingGlobals.h"
using namespace GigaEntity;

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
	Timer t;
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
	t.start();
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

	auto elapsed = t.stop("test");
	auto gb = 1024 * 1024 * 1024;
	std::cout << totalCount * sizeof(long long) * (1000.0 / elapsed) / gb * times << std::endl;
	for (auto armyId = 0; armyId < armyCount; armyId++)
	{
		std::cout << armiesResults[armyId] << std::endl;
	}
}

int main()
{
	// ArrTest();
	// return 0;
	HINSTANCE hInstance = (HINSTANCE)GetModuleHandle(NULL);
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	try
	{
		InitDirect3DApp theApp(hInstance);
		if (!theApp.Initialize())
			return 0;

		return theApp.Run();
	}
	catch (DxException& e)
	{
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
	return 0;
	CudaTest();
	Timer timer(false);
	auto manager = EntityManager();
	manager.AddType<int>();
	manager.AddType<float>();
	auto buffer = CommandBuffer();
	buffer.RegisterComponent<int>();
	constexpr auto count = DEBUG ? 5000 * 100 : 5000 * 10000;
	timer.start();
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
	timer.stop("Tasks");
	timer.start();
	manager.ExecuteCommands(buffer);
	timer.stop("Execute");
	auto array1 = manager.GetComponentArray<int>();
	timer.start();

	auto system = EcsSystem(manager);
	auto newBuffer = CommandBuffer();
	newBuffer.RegisterComponent<float>();
	auto args = ArgumentsObject(newBuffer, newBuffer.GetFastAddHandle<float>());
	auto builder = system.Builder().WithEntities<int>().WithArguments(args);
	auto sys = builder.WithFunction < CastFunction(LambdaFunc) >();
	sys.Run();
	manager.ExecuteCommands(newBuffer);
	timer.stop("System");

	for (int i = 0; i < count; i++)
	{
		auto val1 = array1[i];
		if (array1[i] != i + i)
		{
			throw "false";
		}
	}
}
