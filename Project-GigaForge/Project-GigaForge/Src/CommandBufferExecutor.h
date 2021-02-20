#pragma once
#include "CommandBuffer.h"
#include "EntityManager.h"
#include "../Globals.h"

namespace GigaEntity
{
	class CommandBufferExecutor
	{
	public:
		template <typename T>
		void ExecuteAddStream(std::any commandsAny, std::any componentArrayAny)
		{
			auto& commands = std::any_cast<ConcurrentVector<AddCommand<T>>&>(commandsAny);
			auto& componentArray = std::any_cast<ComponentArray<T>&>(componentArrayAny);
#pragma omp parallel for schedule(static, 1024)
			for (int i = 0; i < commands.Size(); i++)
			{
				auto command = commands[i];
				componentArray.Set(command.entityId, command.item);
			}
		}
	};
}
