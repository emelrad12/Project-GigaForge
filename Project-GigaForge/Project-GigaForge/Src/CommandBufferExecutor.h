#pragma once
#include "CommandBuffer.h"
#include "EntityManager.h"
#include "../Globals.h"

class CommandBufferExecutor
{
public:
	void ExecuteAddStreams(CommandBuffer buffer, EntityManager manager)
	{
		for (auto kvp : callbacks)
		{
			auto steam = buffer.addStreams[kvp.first];
			
		}
	}


	template <typename T>
	void AddCallback(void (*callback)(void*))
	{
		auto name = typeid(T).name();
		callbacks[name] = callback;
	}

	unordered_map<string, void(*)(void*)> callbacks;
};
