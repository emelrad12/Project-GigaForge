#include <thread>
#include "Timer.h"
#include "cassert"
#include "Src/Ecs/EcsSystem.h"
#include "Src/Ecs/EntityManager.h"
using namespace GigaEntity;

template <typename T>
void Print(T data)
{
	std::cout << data << std::endl;
}

void CudaTest();

#define ompLoop omp parallel for schedule(static, 5000 * 100)

void LambdaFunc (int entityIndex, int& item, std::tuple<> arguments)
{
	if (item > entityIndex)
	{
		item--;
	}
	else
	{
		item++;
	}
};

int main()
{
	CudaTest();
	Timer timer(false);
	auto manager = EntityManager();
	manager.AddType<int>();
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
	int m = 50;
	auto tu = tuple(std::ref(m));
	auto& ref1 = std::get<0>(tu);
	ref1++;

	auto system = EcsSystem(manager);
	
	auto builder = system.Builder().WithEntities<int>();
	auto sys = builder.WithFunction<reinterpret_cast<void(*)()>(LambdaFunc)>();
	sys.Run();
	sys.Run();
	timer.stop("System");

	for (int i = 0; i < count; i++)
	{
		auto val1 = array1[i];
		if (array1[i] != i)
		{
			throw "false";
		}
	}
}
