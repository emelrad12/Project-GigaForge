#pragma once
#include "../Globals.h"

const int chunkSize = 1024 * 256;
const int chunkCount = 1024 * 4;

constexpr auto globalCount = DEBUG ? 5000 * 100 : 5000 * 10000;