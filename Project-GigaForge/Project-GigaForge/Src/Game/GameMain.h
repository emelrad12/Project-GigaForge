#pragma once
#include "../Globals.h"
#include "string"
using namespace std::literals::string_literals;

class GameMain
{
	int tick;
public:
	void Start()
	{
		// InitGameWorld();
		// InitRender();
	}

	void Update()
	{
		Print("Run tick "s + std::to_string(tick));
		// UpdateGameWorld();
		// UpdateRender();
		tick++;
	}
};