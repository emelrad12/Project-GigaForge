#pragma once
#include <unordered_map>
#include <iostream>
#include <vector>
#include <unordered_map>

template <typename T>
void Print(T data)
{
	std::cout << data << std::endl;
}

using std::vector;
using std::string;
using std::unordered_map;
using std::tuple;
#define concurrentVectorImpl ConcurrentVector
#ifdef _DEBUG
#define DEBUG true
#else
#define DEBUG false
#endif