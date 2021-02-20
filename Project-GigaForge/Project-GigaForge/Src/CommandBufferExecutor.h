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
			CombinedChunk<T> currentChunk = CombinedChunk<T>(-1, 0, nullptr, nullptr);
			auto size = commands.size();
			for (int i = 0; i < size; i++)
			{
				auto command = commands[i];
				auto entityId = command.entityId;
				if (currentChunk.end <= 0 || entityId >= currentChunk.end || entityId < currentChunk.start)
				{
					currentChunk = componentArray.GetCombinedChunk(entityId);
				}

				if (currentChunk.end == 0)
				{
					componentArray.AllocateChunk(entityId);
					currentChunk = componentArray.GetCombinedChunk(entityId);
				}

				if (currentChunk.end <= 0 || entityId >= currentChunk.end)
				{
					throw std::exception("Failed to allocate");
				}
				componentArray.SetAt(currentChunk, entityId, command.item);
			}
		}
	};
}
