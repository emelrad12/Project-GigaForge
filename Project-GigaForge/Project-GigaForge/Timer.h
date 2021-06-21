#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <utility>

class Timer
{
public:
	void Start()
	{
		begin = std::chrono::high_resolution_clock::now();
	}

	int Stop(std::string name)
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		long long ms;
		if (useMicro)
		{
			ms = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
		}
		else
		{
			ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		}
		std::cout << name << ": " << ms << std::endl;
		return ms;
	}

	int Restart(std::string name)
	{
		const auto data = Stop(std::move(name));
		Start();
		return data;
	}

	Timer(bool useMicro = false): useMicro(useMicro)
	{
		Start();
	}

private:
	bool useMicro;
	std::chrono::steady_clock::time_point begin;
};
