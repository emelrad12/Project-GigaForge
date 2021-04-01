#pragma once
#include <any>
#include "../Ecs/EntityManager.h"
#include "unordered_set"

namespace GigaEntity
{
	class CudaEntityManager
	{
	public:
		CudaEntityManager(EntityManager entityManager) : entityManager(entityManager)
		{
		}

		template <typename T>
		void CopyToCuda()
		{
			auto name = typeid(T).name();
			types.emplace(name);
			data[name] = std::any(CudaComponentArray<T>(entityManager.GetComponentArray<T>()));
		}

		template <typename T>
		CudaComponentArray<T> GetComponentArray()
		{
			auto name = typeid(T).name();
			return std::any_cast<CudaComponentArray<T>>(data[name]);
		}

		std::unordered_set<string> types = std::unordered_set<string>();
		unordered_map<string, std::any> data = unordered_map<string, std::any>();
		int itemCount = DEBUG ? 5000 * 100 : 5000 * 10000; //todo
		EntityManager entityManager;
	};
}
