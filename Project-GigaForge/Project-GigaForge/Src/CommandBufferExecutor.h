#pragma once
#include "CommandBuffer.h"
#include "EntityManager.h"
#include "../Globals.h"
#include "boost/any.hpp"

namespace GigaEntity
{
	class CommandBufferExecutor
	{
	public:
		template <typename T>
		void ExecuteAddStream(std::any commandsAny, std::any componentArrayAny)
		{
			auto& commands = std::any_cast<concurrent_vector<AddCommand<T>>&>(commandsAny);
			auto& componentArray = std::any_cast<ComponentArray<T>&>(componentArrayAny);
			for (auto command : commands)
			{
				componentArray.Set(command.entityId, command.item);
			}
		}
	};
}
