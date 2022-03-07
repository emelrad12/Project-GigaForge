#pragma once
#include "doctest.h"
#include "../Src/Game/Traffic/Lane.h"
#include "Timer.hpp"
using namespace GigaForge;
TEST_CASE("Test Traffic simple")
{
	Init();
	auto lane = Lane();
	auto car = LaneVehicle();
	for (size_t i = 0; i < 10; i++)
	{
		lane.AddToEnd(car);
	}

	for (int i = 0; i < 100; i++)
	{
		lane.Iterate();
	}
}

TEST_CASE("Test Traffic massive")
{
	Init();
	auto laneCount = 100*100;
	auto iterations = 1000;
	auto vehiclesPerLane = 100;
	auto lanes = vector<Lane>(laneCount);
	auto car = LaneVehicle();
	for (int i = 0; i < laneCount; i++)
	{
		lanes[i] = Lane();
		for (int j = 0; j < vehiclesPerLane; j++)
		{
			auto& lane = lanes[i];
			lane.AddToEnd(car);
		}
	}

	auto timer = Diligent::Timer();
#pragma omp parallel for
	for (int i = 0; i < laneCount; i++)
	{
		for (int j = 0; j < iterations; j++)
		{
			auto& lane = lanes[i];
			lane.Iterate();
		}
	}
	std::cout << "Time per iteration: " << timer.GetElapsedTime() * 1000 / iterations << std::endl;
	std::cout << "Total time: " << timer.GetElapsedTime() * 1000 << std::endl;
	std::cout << "Total simulated vehicles: " << laneCount * iterations * vehiclesPerLane / (1000 * 1000) << "M" << std::endl;
}
