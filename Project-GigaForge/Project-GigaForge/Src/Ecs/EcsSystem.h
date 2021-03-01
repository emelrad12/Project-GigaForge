#pragma once
#include <functional>

#include "EntityManager.h"

namespace GigaEntity
{
	class EcsSystem
	{
	public:
		explicit EcsSystem(EntityManager& manager)
			: manager(manager)
		{
		}

		EntityManager& manager;

		template <typename ...ArgumentTypes>
		struct WithArgumentsStruct
		{
			WithArgumentsStruct(EntityManager& manager)
				: manager(manager)
			{
			}

			EntityManager& manager;

			template <typename T1, void (*functionPointer)(int, T1&, ArgumentTypes ...)>
			void EntitiesForeach(ArgumentTypes ... items)
			{
				auto arr1 = manager.GetComponentArray<T1>();
				for (int i = 0; i < manager.itemCount; i++)
				{
					if (arr1.ContainsEntity(i))
					{
						functionPointer(i, arr1[i], items...);
					}
				}
			}

			template <typename T1, typename T2, void (*functionPointer)(int, T1&, T2&, ArgumentTypes ...)>
			void EntitiesForeach(ArgumentTypes ... items)
			{
				auto arr1 = manager.GetComponentArray<T1>();
				auto arr2 = manager.GetComponentArray<T2>();
				for (int i = 0; i < manager.itemCount; i++)
				{
					if (arr1.ContainsEntity(i) && arr2.ContainsEntity(i))
					{
						functionPointer(i, arr1[i], arr2[i], items...);
					}
				}
			}
		};

		template <typename ...Types>
		WithArgumentsStruct<Types...> WithArguments()
		{
			return WithArgumentsStruct<Types...>(manager);
		}

	private:
	};
}
