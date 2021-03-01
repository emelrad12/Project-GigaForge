#pragma once
#include <any>
#include "Globals.h"
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
		T* items;
		CopyableAtomic<int> currentIndex;
	public:
		ConcurrentVector()
		{
			constexpr auto count = DEBUG ? 5000 * 100 : 5000 * 10000;//todo
			items = new T[count];
		}

		void push_back(T item)
		{
			auto index = currentIndex++;
			items[index] = item;
		}

		T& operator [](int index)
		{
			auto a = T();
			return items[index];
		}

		int size() const
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
			const string name = typeid(T).name();
			deleteComponentStreams[name] = concurrentVectorImpl<int>();
			addStreams[name] = concurrentVectorImpl<AddCommand<T>>();
		}

		template <typename T>
		void AddComponent(int entity, T item, concurrentVectorImpl<AddCommand<T>>& handle)
		{
			handle.push_back(AddCommand<T>{item, entity});
		}

		template <typename T>
		concurrentVectorImpl<AddCommand<T>>& GetFastAddHandle()
		{
			const auto name = typeid(T).name();
			return std::any_cast<concurrentVectorImpl<AddCommand<T>>&>(addStreams[name]);
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

		concurrentVectorImpl<int> deleteEntityStream;
		unordered_map<string, concurrentVectorImpl<int>> deleteComponentStreams;
		unordered_map<string, std::any> addStreams;
	};
}
