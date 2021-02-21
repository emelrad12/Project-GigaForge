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

#define ompLoop omp parallel for schedule(static, 5000 * 100)

int main()
{
	Timer timer("Taken: ");
	GigaEntity::EntityManager manager = GigaEntity::EntityManager();
	manager.AddType<int>();
	manager.AddType<double>();
	manager.AddType<bool>();
	manager.AddType<short>();
	manager.AddType<unsigned>();
	manager.AddType<long long>();
	manager.AddType<uint8_t>();
	manager.AddType<int8_t>();
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
	auto task2 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<double>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<double>(index, index, handle);
				}
			}
		}
	};
	auto task3 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<bool>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<bool>(index, true, handle);
				}
			}
		}
	};
	auto task4 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<short>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<short>(index, true, handle);
				}
			}
		}
	};
	auto task5 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<unsigned>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<unsigned>(index, true, handle);
				}
			}
		}
	};
	auto task6 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<long long>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<long long>(index, true, handle);
				}
			}
		}
	};
	auto task7 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<uint8_t>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<uint8_t>(index, true, handle);
				}
			}
		}
	};
	auto task8 = [&buffer]()
	{
		auto& handle = buffer.GetFastAddHandle<int8_t>();
		for (int offset = 0; offset < count; offset += chunkSize)
		{
#pragma ompLoop
			for (int i = 0; i < chunkSize; i++)
			{
				const auto index = i + offset;
				if (index < count)
				{
					buffer.AddComponent<int8_t>(index, true, handle);
				}
			}
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
	timer.stop();
	timer.start();
	manager.ExecuteCommands(buffer);
	timer.stop();
	auto array1 = manager.GetComponentArray<int>();
	auto array2 = manager.GetComponentArray<double>();
	auto array3 = manager.GetComponentArray<bool>();
	for (int i = 0; i < count; i++)
	{
		auto val1 = array1[i];
		auto val2 = array2[i];
		auto val3 = array3[i];
		if (array1[i] != i || array2[i] != i || array3[i] != true)
		{
			throw "false";
		}
	}
}
