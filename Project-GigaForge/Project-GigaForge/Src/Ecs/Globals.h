#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
using std::vector;
using std::string;
using std::unordered_map;
using std::tuple;
const int chunkSize = 1024 * 256;
const int chunkCount = 1024 * 4;
#define concurrentVectorImpl ConcurrentVector
#ifdef _DEBUG
#define DEBUG true
#else
#define DEBUG false
#endif
constexpr auto globalCount = DEBUG ? 50 * 1 : 5000 * 10000;