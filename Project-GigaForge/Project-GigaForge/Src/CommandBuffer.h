#pragma once
#include <any>
#include "../Globals.h"
#include <concurrent_vector.h>

#include "CopyableAtomic.h"
using concurrency::concurrent_vector;

namespace GigaEntity
{	
	template <typename T>
	struct AddCommand
	{
		T item;
		int entityId;
	};

	template <typename T>
	class ConcurrentVector
	{
		vector<T> items;
		CopyableAtomic<int> currentIndex;
	public:
		ConcurrentVector()
		{
			items = vector<T>(5000 * 10000);
		}
		
		void Add(T item)
		{
			auto index = currentIndex++;
			items[index] = item;
		}

		T& operator [](int index)
		{
			auto a = T();
			return items[index];
		}

		int Size() const
		{
			return currentIndex;
		}
	};

	class CommandBuffer
	{
	public:
		template <typename T>
		void RegisterComponent()
		{
			auto vecot = ConcurrentVector<T>();
			const string name = typeid(T).name();
			deleteComponentStreams[name] = ConcurrentVector<int>();
			addStreams[name] = ConcurrentVector<AddCommand<T>>();
		}

		template <typename T>
		void AddComponent(int entity, T item, ConcurrentVector<AddCommand<T>>& handle)
		{
			handle.Add(AddCommand<T>{item, entity});
		}

		template <typename T>
		ConcurrentVector<AddCommand<T>>& GetFastAddHandle()
		{
			const auto name = typeid(T).name();
			return std::any_cast<ConcurrentVector<AddCommand<T>>&>(addStreams[name]);
		}

		template <typename T>
		void RemoveComponent(int entityId)
		{
			const auto name = typeid(T).name();
			deleteComponentStreams[name].Add(entityId);
		}

		void DeleteEntity(int entityId)
		{
			deleteEntityStream.Add(entityId);
		}

		ConcurrentVector<int> deleteEntityStream;
		unordered_map<string, ConcurrentVector<int>> deleteComponentStreams;
		unordered_map<string, std::any> addStreams;
	};
}
