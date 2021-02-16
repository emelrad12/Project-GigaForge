#pragma once
#include <any>
#include "../Globals.h"
#include "ppl.h"
#include <concurrent_vector.h>
using concurrency::concurrent_vector;

namespace GigaEntity
{
	template <typename T>
	struct AddCommand
	{
		T item;
		int entityId;
	};

	class CommandBuffer
	{
	public:
		template <typename T>
		void RegisterComponent()
		{
			const auto name = typeid(T).name();
			deleteComponentStreams[name] = concurrent_vector<int>();
			auto data = concurrent_vector<AddCommand<T>>();
			data.reserve(5000 * 1000);
			addStreams[name] = data;
		}

		template <typename T>
		void AddComponent(int entity, T item, concurrent_vector<AddCommand<T>>& handle)
		{
			handle.push_back(AddCommand<T>{item, entity});
		}
		
		template <typename T>
		concurrent_vector<AddCommand<T>>& GetFastAddHandle()
		{
			const auto name = typeid(T).name();
			return std::any_cast<concurrent_vector<AddCommand<T>>&>(addStreams[name]);
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
}
