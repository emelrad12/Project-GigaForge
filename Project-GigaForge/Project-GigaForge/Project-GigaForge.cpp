#include "Globals.h"
#include <thread>
#include "Timer.h"
#include "Src/ComponentArray.h"
#include "Src/EntityManager.h"
#include "cassert"

template <typename T>
void Print(T data)
{
	std::cout << data << std::endl;
}


int main()
{
	Timer timer("Taken: ");
	GigaEntity::EntityManager manager = GigaEntity::EntityManager();
	manager.AddType<int>();
	manager.AddType<double>();
	manager.AddType<bool>();
	GigaEntity::CommandBuffer buffer = GigaEntity::CommandBuffer();
	buffer.RegisterComponent<int>();
	buffer.RegisterComponent<double>();
	buffer.RegisterComponent<bool>();
	constexpr auto count = DEBUG ? 5000 * 100 : 5000 * 10000;
	timer.start();
	auto task1 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<int>();
#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<int>(i, i, handle);
		}
	};
	auto task2 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<double>();
#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<double>(i, i, handle);
		}
	};
	auto task3 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<bool>();
#pragma omp parallel for
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<bool>(i, true, handle);
		}
	};

	std::thread t1(task1);
	std::thread t2(task2);
	std::thread t3(task3);

	t1.join();
	t2.join();
	t3.join();
	manager.ExecuteCommands(buffer);
	timer.stop();
	auto array1 = manager.GetComponentArray<int>();
	auto array2 = manager.GetComponentArray<double>();
	auto array3 = manager.GetComponentArray<bool>();
	for (int i = 0; i < count; i++)
	{
		assert(array1[i] == (int)i);
		assert(array2[i] == (double)i);
		assert(array3[i] == true);
	}
}
