#include <thread>
#include "Timer.h"
#include "cassert"
#include "Src/Ecs/EcsSystem.h"
#include "Src/Ecs/EntityManager.h"
#include "Src/Ecs/CommandBuffer.h"
#include "Src/Rendering/RenderingMain.h"
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
	ArgumentsObject(CommandBuffer buffer, FastHandle(float) floatHandle): buffer(buffer), floatHandle(floatHandle)
	{
	}

	CommandBuffer& buffer;
	FastHandle(float)& floatHandle;
};

void LambdaFunc(int entityIndex, int& item, ArgumentsObject arguments)
{
	item += entityIndex;
}

int main()
{
	Rendering::Renderer r = Rendering::Renderer();
	r.Init();
	r.Run();
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
	auto sys = builder.WithFunction < CastFunction(LambdaFunc) > ();
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
