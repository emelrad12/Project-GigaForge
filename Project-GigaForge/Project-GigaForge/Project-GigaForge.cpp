#define NOMINMAX true
#define D3D12_SUPPORTED true
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Src/Globals.h"
#include <wrl/module.h>
#include "Timer.h"
#include "cassert"
#include "Src/Ecs/CommandBuffer.h"
#include "Src/Game/GameMain.h"
using namespace GigaEntity;

int MainTest()
{
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
	auto testLoops = 1000;
	auto game = GameMain();
	game.Start();
	while (testLoops-- > 0)
	{
		game.Update();
	}
	return 0;
}

TEST_CASE("Main")
{
	MainTest();
}
