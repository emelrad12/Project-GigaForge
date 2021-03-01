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

#define ompLoop omp parallel for schedule(static, 5000 * 100)

int main()
{
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

	auto system = EcsSystem(manager);
	auto offset = 55;
	constexpr auto func = [](int index, int& item, int offset)
	{
		if (item + offset != index + offset)
		{
			item--;
		}
		else
		{
			item++;
		}
	};

	system.WithArguments<int>().EntitiesForeach<int, func>(offset);
	system.WithArguments<int>().EntitiesForeach<int, func>(offset);

	for (int i = 0; i < count; i++)
	{
		auto val1 = array1[i];
		if (array1[i] != i)
		{
			throw "false";
		}
	}
}
