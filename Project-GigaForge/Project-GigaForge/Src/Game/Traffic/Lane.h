#pragma once
#include "VehicleTemplate.h"
#include "../../Globals.h"

namespace GigaForge
{
	inline vector<VehicleTemplate> templates = vector<VehicleTemplate>(1);

	inline void Init()
	{
		templates[0] = {5, 1, 1};
	}

	class LaneVehicle
	{
	public:
		float distanceFromStart;
		float vehicleSpeed; //Todo optimize
		int templateId;
		int vehicleData;
	};

	class Lane
	{
	public:
		Lane() : vehicles(vector<LaneVehicle>(10)), beginningId(0), lenId(0), length(5000)
		{
		}

		std::vector<LaneVehicle> vehicles;
		int beginningId;
		int lenId;
		float length;

		LaneVehicle RemoveFirst()
		{
			lenId--;
			return vehicles[beginningId++];
		}

		void AddToEnd(LaneVehicle& vehicle)
		{
			auto newLen = beginningId + lenId++;
			if (newLen >= vehicles.capacity()) vehicles.resize(newLen + 1);
			vehicle.distanceFromStart = length;
			vehicles[newLen] = vehicle;
		}

		void Iterate()
		{
			for (int i = 0; i < lenId; i++)
			{
				const auto currentIndex = beginningId + i;
				auto& currentVehicle = vehicles[currentIndex];
				const auto& vehicleTemplate = templates[currentVehicle.templateId];
				if (currentVehicle.vehicleSpeed < vehicleTemplate.maxSpeed)
				{
					currentVehicle.vehicleSpeed += vehicleTemplate.acceleration;
					if (currentVehicle.vehicleSpeed > vehicleTemplate.maxSpeed)currentVehicle.vehicleSpeed = vehicleTemplate.maxSpeed;
				}
				currentVehicle.distanceFromStart -= currentVehicle.vehicleSpeed;
			}

			for (int i = 0; i < lenId; i++)
			{
				const auto currentIndex = beginningId + i;
				auto& currentVehicle = vehicles[currentIndex];
				const auto& vehicleTemplate = templates[currentVehicle.templateId];
				auto& currentDistanceFromStart = currentVehicle.distanceFromStart;
				float nextCarDistanceFromStart = 0;
				if (currentIndex > 0)
				{
					const auto& nextCar = vehicles[currentIndex - 1];
					nextCarDistanceFromStart = nextCar.distanceFromStart;
				}
				const auto minDistanceFromStart = nextCarDistanceFromStart + vehicleTemplate.length;
				if (minDistanceFromStart > currentDistanceFromStart)currentDistanceFromStart = minDistanceFromStart;
			}
		}
	};
}
