#pragma once
#include "doctest.h"
#include "../Src/Rendering/RenderingTestWin32.h"
#include "../Src/Game/GameMain.h"
#include <Windows.h>

TEST_CASE("Test Rendering")
{
	HINSTANCE hInstance = (HINSTANCE)GetModuleHandle(NULL);
	RenderInit(hInstance, 1);
	auto game = GameMain();
	game.Start();
	while (true)
	{
		game.Update();
	}
}
