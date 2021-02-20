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

#define ompLoop omp parallel for
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
	buffer.RegisterComponent<short>();
	buffer.RegisterComponent<unsigned>();
	buffer.RegisterComponent<long long>();
	buffer.RegisterComponent<uint8_t>();
	buffer.RegisterComponent<int8_t>();
	constexpr auto count = DEBUG ? 5000 * 100 : 5000 * 10000;
	const int scheduleNum = 1024 * 1024;
	timer.start();
	auto task1 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<int>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<int>(i, i, handle);
		}
	};
	auto task2 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<double>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<double>(i, i, handle);
		}
	};
	auto task3 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<bool>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<bool>(i, true, handle);
		}
	};
	auto task4 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<short>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<short>(i, true, handle);
		}
	};
	auto task5 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<unsigned>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<unsigned>(i, true, handle);
		}
	};
	auto task6 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<long long>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<long long>(i, true, handle);
		}
	};
	auto task7 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<uint8_t>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<uint8_t>(i, true, handle);
		}
	};
	auto task8 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<int8_t>();
#pragma ompLoop
		for (int i = 0; i < count; i++)
		{
			buffer.AddComponent<int8_t>(i, true, handle);
		}
	};

	std::thread t1(task1);
	std::thread t2(task2);
	std::thread t3(task3);
	std::thread t4(task4);
	std::thread t5(task5);
	std::thread t6(task6);
	std::thread t7(task7);
	std::thread t8(task8);

	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();
	t8.join();
	manager.ExecuteCommands(buffer);
	timer.stop();
	auto array1 = manager.GetComponentArray<int>();
	auto array2 = manager.GetComponentArray<double>();
	auto array3 = manager.GetComponentArray<bool>();
	for (int i = 0; i < count; i++)
	{
		if (array1[i] != i || array2[i] != i || array3[i] != true)
		{
			throw "false";
		}
	}
}
