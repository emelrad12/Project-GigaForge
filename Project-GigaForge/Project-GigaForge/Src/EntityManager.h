#pragma once
#include "../Globals.h"
#include <any>
#include "ComponentArray.h"

class EntityManager
{
public:
	template <typename T>
	void AddType()
	{
		auto name = typeid(T).name();
		types.push_back(name);
		data[name] = std::any(ComponentArray<T>(1024, 1024));
	}

	template <typename T>
	ComponentArray<T> GetComponentArray()
	{
		auto name = typeid(T).name();
		return std::any_cast<ComponentArray<T>>(data[name]);
	}

	unordered_map<string, std::any> data = unordered_map<string, std::any>();

	vector<string> types = vector<string>();
};
