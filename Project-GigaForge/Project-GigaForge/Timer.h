#pragma once
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
public:
	void start()
	{
		begin = std::chrono::high_resolution_clock::now();
	}

	void stop()
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
	}

	Timer(std::string name, bool useMicro = false): useMicro(useMicro)
	{
		Timer::name = name;
	}

private:
	std::string name;
	bool useMicro;
	std::chrono::steady_clock::time_point begin;
};
