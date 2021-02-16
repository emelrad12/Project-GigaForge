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
			for (size_t i = 0; i < commands.size(); i++)
			{
				auto command = commands[i];
				componentArray.Set(command.entityId, command.item);
			}
		}
	};
}
