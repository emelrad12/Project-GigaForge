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

	float Stop(std::string name)
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		float time = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
		if (!useMicro)
		{
			time /= 1000;
		}
		
		std::cout << name << ": " << time << std::endl;
		return time;
	}

	float Restart(std::string name)
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
