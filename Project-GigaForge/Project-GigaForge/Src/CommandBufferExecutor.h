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
				if (currentChunk.end <= 0 || i >= currentChunk.end)
				{
					currentChunk = componentArray.GetCombinedChunk(i);
				}

				if (currentChunk.end == 0)
				{
					componentArray.AllocateChunk(i);
					currentChunk = componentArray.GetCombinedChunk(i);
				}

				if (currentChunk.end <= 0 || i >= currentChunk.end)
				{
					throw std::exception("Failed to allocate");
				}
				auto command = commands[i];
				componentArray.SetAt(currentChunk, command.entityId, command.item);
			}
		}
	};
}
