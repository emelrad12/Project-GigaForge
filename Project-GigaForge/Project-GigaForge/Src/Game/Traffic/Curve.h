#pragma once
#include "eigen/Core"

class Curve
{
	Eigen::Vector2f p1;
	Eigen::Vector2f p2;
	Eigen::Vector2f p3;

	bool Intersect(Curve otherCurve, Eigen::Vector2f& output)
	{
		auto x1 = p1.x();
		auto x2 = p2.x();
		auto y1 = p1.y();
		auto y2 = p2.y();
		auto x3 = otherCurve.p1.x();
		auto x4 = otherCurve.p2.x();
		auto y3 = otherCurve.p1.y();
		auto y4 = otherCurve.p2.y();
		auto x12 = x1 - x2;
		auto x34 = x3 - x4;
		auto y12 = y1 - y2;
		auto y34 = y3 - y4;

		auto c = x12 * y34 - y12 * x34;

		if (fabs(c) < 0.01f)
		{
			// No intersection
			return false;
		}
		else
		{
			// Intersection
			auto a = x1 * y2 - y1 * x2;
			auto b = x3 * y4 - y3 * x4;

			auto x = (a * x34 - b * x12) / c;
			auto y = (a * y34 - b * y12) / c;
			output.x() = x;
			output.y() = y;
			return true;
		}
	}
};
