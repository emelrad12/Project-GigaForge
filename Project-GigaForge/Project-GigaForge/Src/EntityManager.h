#pragma once
#include "../Globals.h"
#include <any>

#include "CommandBuffer.h"
#include "CommandBufferExecutor.h"
#include "ComponentArray.h"

namespace GigaEntity
{
	class EntityManager
	{
	public:
		template <typename T>
		void AddType()
		{
			auto name = typeid(T).name();
			types.emplace_back(name);
			data[name] = std::any(ComponentArray<T>(chunkCount, chunkSize));
			addComponentExecutorFunction[name] = &CommandBufferExecutor::ExecuteAddStream<T>;
		}

		template <typename T>
		ComponentArray<T> GetComponentArray()
		{
			auto name = typeid(T).name();
			return std::any_cast<ComponentArray<T>>(data[name]);
		}

		void ExecuteCommands(CommandBuffer& commandBuffer)
		{
			#pragma omp parallel for
			for (int i = 0; i < types.size(); i++)
			{
				auto type = types[i];
				(executor.*addComponentExecutorFunction[type])(commandBuffer.addStreams[type], data[type]);
			}
		}

		vector<string> types = vector<string>();
		CommandBufferExecutor executor = CommandBufferExecutor();
		unordered_map<string, std::any> data = unordered_map<string, std::any>();
		unordered_map<string, void(CommandBufferExecutor::*)(std::any commandsAny, std::any componentArrayAny)>
		addComponentExecutorFunction = unordered_map<string, void(CommandBufferExecutor::*)(
			                                             const std::any commandsAny, std::any componentArrayAny)>();
	};
}
