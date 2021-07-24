#pragma once
#include <cstdint>
#include <boost/functional/hash.hpp>
namespace GigaEntity
{
	struct DropletHandle
	{
		uint32_t id;

		explicit DropletHandle(const int newId) : id(newId)
		{
		}

		friend size_t hash_value(const DropletHandle& v) {
			using boost::hash_value;
			return hash_value(v.id);
		}

		friend bool operator==(const DropletHandle& lhs, const DropletHandle& rhs) {
			return hash_value(lhs) == hash_value(rhs);
		}

	};
	

	struct DropletMapping
	{
		uint16_t dropletId;
		uint32_t offset;
		uint16_t length;
	};
}
template <> struct std::hash<GigaEntity::DropletHandle> : boost::hash<GigaEntity::DropletHandle> {};
