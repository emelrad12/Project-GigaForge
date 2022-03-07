#pragma once
#include "CommandBuffer.h"
#include "ComponentArray.h"
#include "../Globals.h"

namespace GigaEntity
{
	class CommandBufferExecutor
	{
	public:
		template <typename T>
		void ExecuteAddStream(std::any commandsAny, std::any componentArrayAny)
		{
			int hits = 0;
			int misses = 0;
			auto& commands = std::any_cast<concurrentVectorImpl<AddCommand<T>>&>(commandsAny);
			auto& componentArray = std::any_cast<ComponentArray<T>&>(componentArrayAny);
			auto currentChunk = CombinedChunk<T>::GetDefault();
			auto previous = CombinedChunk<T>::GetDefault();
			auto size = commands.size();
			for (int i = 0; i < size; i++)
			{
				auto command = commands[i];
				auto entityId = command.entityId;
				if (currentChunk.end <= 0 || entityId >= currentChunk.end || entityId < currentChunk.start)
				{
					misses++;
					if (entityId < previous.end && entityId >= previous.start)
					{
						hits++;
						std::swap(currentChunk, previous);
					}
					else
					{
						previous = currentChunk;
						currentChunk = componentArray.GetCombinedChunk(entityId);
					}
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
			std::cout << "Hits: " << hits << "  Misses: " << misses << "  Ratio: " << (float)hits / misses << std::endl;
		}
	};
}
