#pragma once
#include <concurrent_vector.h>
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
using Concurrency::concurrent_vector;
#define concurrentVectorImpl concurrent_vector
#ifdef _DEBUG
#define DEBUG true
#else
#define DEBUG false
#endif