#pragma once
#include "EntityManager.h"
using std::tuple;
void VoidFunc();

#define CastFunction(func) reinterpret_cast<void(*)()>(func)

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

		template <class TEntities, class TArguments, void (*TFunctionPointer)()>
		struct FunctionBuilder
		{
			FunctionBuilder(EntityManager& manager) : manager(manager)
			{
			}

			FunctionBuilder(EntityManager& manager, TArguments newArguments) : manager(manager), arguments(newArguments)
			{
			}

			template <class TNewArguments>
			FunctionBuilder<TEntities, TNewArguments, TFunctionPointer> WithArguments(TNewArguments newArgs)
			{
				return FunctionBuilder<TEntities, TNewArguments, TFunctionPointer>(manager, newArgs);
			}

			template <typename... TNewEntities>
			FunctionBuilder<tuple<TNewEntities...>, TArguments, TFunctionPointer> WithEntities()
			{
				return FunctionBuilder<tuple<TNewEntities...>, TArguments, TFunctionPointer>(manager, arguments);
			}

			template <void (*TNewFunctionPointer)()>
			FunctionBuilder<TEntities, TArguments, TNewFunctionPointer> WithFunction()
			{
				return FunctionBuilder<TEntities, TArguments, TNewFunctionPointer>(manager, arguments);
			}

			void Run()
			{
				Foreach(TEntities());
			}

			template <typename E1>
			void Foreach(tuple<E1>)
			{
				auto arr1 = manager.GetComponentArray<E1>();
#pragma omp parallel for
				for (int i = 0; i < manager.itemCount; i++)
				{
					if (arr1.ContainsEntity(i))
					{
						Call(reinterpret_cast<void (*)()>(TFunctionPointer), i, arr1[i]);
					}
				}
			}

			template <typename ... EntityTypes>
			void Call(void (*f)(), int index, EntityTypes&... entities)
			{
				auto castPointer = reinterpret_cast<void(*)(int, EntityTypes& ..., TArguments)>(f);
				castPointer(index, entities..., arguments);
			}

			TArguments arguments;
			EntityManager& manager;
		};

		FunctionBuilder<tuple<>, tuple<>, reinterpret_cast<void(*)()>(VoidFunc)> Builder()
		{
			auto builder = FunctionBuilder<tuple<>, tuple<>, reinterpret_cast<void(*)()>(VoidFunc)>(manager);
			return builder;
		}
	};
}
