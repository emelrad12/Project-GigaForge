#pragma once
#include "doctest.h"
#include "../Src/OptixMesh/Mesher.h"

TEST_CASE("Mesher Test")
{
	auto mesher = Mesher();
	mesher.Start();
}