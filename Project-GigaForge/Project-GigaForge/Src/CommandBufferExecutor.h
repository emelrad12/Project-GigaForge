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
			int validUntil = -1;
			for (int i = 0; i < commands.size(); i++)
			{
				if (validUntil <= 0 || i >= validUntil)
				{
					validUntil = componentArray.GetValidChunkUntil(i);
				}
				if (validUntil == 0)
				{
					componentArray.AllocateChunk(i);
					validUntil = componentArray.GetValidChunkUntil(i);
				}
				if(validUntil <= 0 || i >= validUntil)
				{
					throw std::exception("Failed to allocate");
				}
				auto command = commands[i];
				componentArray.Set(command.entityId, command.item);
			}
		}
	};
}
