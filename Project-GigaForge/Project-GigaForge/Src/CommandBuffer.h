#pragma once
#include <any>
#include "../Globals.h"
#include "ppl.h"
#include <concurrent_vector.h>
using concurrency::concurrent_vector;

class CommandBuffer
{
public:
	template <typename T>
	struct AddCommand
	{
		T item;
		int entityId;
	};

	template <typename T>
	void AddComponent(int entity, T item)
	{
		const auto name = typeid(T).name();
		auto currentArray = std::any_cast<concurrent_vector<AddCommand<T>>>(addStreams[name]);
		currentArray.push_back(AddCommand<T>{item, entity});
	}

	template <typename T>
	void RemoveComponent(int entityId)
	{
		const auto name = typeid(T).name();
		deleteComponentStreams[name].push_back(entityId);
	}

	void DeleteEntity(int entityId)
	{
		deleteEntityStream.push_back(entityId);
	}

	concurrent_vector<int> deleteEntityStream;
	unordered_map<string, concurrent_vector<int>> deleteComponentStreams;
	unordered_map<string, std::any> addStreams;
};
