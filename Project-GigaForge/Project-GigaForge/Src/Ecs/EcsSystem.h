#pragma once
#include "EntityManager.h"
using std::tuple;
void func(int, tuple<tuple<>, tuple<>>);

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

		template <class TEntities, class TArguments, typename TFunctionArgs, void (*TFunctionPointer)(int, TFunctionArgs)>
		struct FunctionBuilder
		{
			FunctionBuilder(EntityManager& manager): manager(manager)
			{
			}

			template <class TNewArguments>
			FunctionBuilder<TEntities, TNewArguments, TFunctionArgs, TFunctionPointer> WithArguments(TArguments newArgs)
			{
				arguments = newArgs;
				return FunctionBuilder<TEntities, TNewArguments, TFunctionArgs, TFunctionPointer>(manager);
			}

			template <class TNewEntities>
			FunctionBuilder<TNewEntities, TArguments, TFunctionArgs, TFunctionPointer> WithEntities()
			{
				return FunctionBuilder<TNewEntities, TArguments, TFunctionArgs, TFunctionPointer>(manager);
			}

			template <void (*TNewFunctionPointer)(int, tuple<TEntities, TArguments>)>
			FunctionBuilder<TEntities, TArguments, tuple<TEntities, TArguments>, TNewFunctionPointer> WithFunction()
			{
				return FunctionBuilder<TEntities, TArguments, tuple<TEntities, TArguments>, TNewFunctionPointer>(manager);
			}

			void Foreach()
			{
				auto t1 = std::get<0>(TEntities());
				auto arr1 = manager.GetComponentArray<decltype(t1)>();
				for (int i = 0; i < manager.itemCount; i++)
				{
					if (arr1.ContainsEntity(i))
					{
						auto entittyArgs = tuple(arr1[0]);
						auto args = tuple(entittyArgs, arguments);
						auto mergedArgs = std::tuple_cat(tuple(i), entittyArgs, tuple(args));
						std::apply(TFunctionPointer,mergedArgs);
						// TFunctionPointer(i, args);
					}
				}
			}

			TArguments arguments;
			EntityManager& manager;
		};

		FunctionBuilder<tuple<>, tuple<>, tuple<tuple<>, tuple<>>, func> Builder()
		{
			auto builder = FunctionBuilder<tuple<>, tuple<>, tuple<tuple<>, tuple<>>, func>(manager);
			return builder;
		}
	};
}
