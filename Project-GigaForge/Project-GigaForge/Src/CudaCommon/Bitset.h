#pragma once

namespace GigaEntity
{
	class Bitset
	{
		const int bitCount = 8;
		/// <summary>
		/// 
		/// </summary>
		/// <param name="unpackedSize">Size in bits, must always be a multiple of 8</param>
	public:
		Bitset(int size) : unpackedSize(size), packedSize((size + bitCount - 1) / bitCount)
		{
			if (size % 8 > 0) throw "Size must be a multiple of 8";
			packedData = new uint8_t[packedSize];
			unpackedData = new bool[size];
		}

		void Unpack()
		{
			isPacked = false;
			unpackedData = new bool[unpackedSize];
			for (int packedIndex = 0; packedIndex < packedSize; packedIndex++)
			{
				const auto word = packedData[packedIndex];
				const auto truePackedIndex = packedIndex * 8;
				for (int bitIndex = 0; bitIndex < bitCount; bitIndex++)
				{
					const auto bit = UnpackFromBits(word, bitIndex);
					unpackedData[truePackedIndex + bitIndex] = bit;
				}
			}
		}

		/// <summary>
		/// Accessing uncompressed data afterwards is not allowed, as it is deallocated
		/// </summary>
		void Pack()
		{
			for (int itemIndex = 0; itemIndex < packedSize; itemIndex += bitCount)
			{
				auto packedIndex = itemIndex / bitCount;
				uint8_t word = 0;
				for (int bitIndex = 0; bitIndex < bitCount; bitIndex++)
				{
					PackBits(word, bitIndex, unpackedData[itemIndex + bitIndex]);
				}
				packedData[packedIndex] = word;
			}
			isPacked = true;
			delete unpackedData;
		}

		[[nodiscard]] bool& At(int location) const
		{
			return unpackedData[location];
		}

		bool isPacked;
		uint8_t* packedData;
		bool* unpackedData;
		int unpackedSize;
		int packedSize;
	
	private:
		static void PackBits(uint8_t& prev, const int location, const bool value)
		{
			if (value)
			{
				prev |= 1 << location;
			}
			else
			{
				prev &= ~(1 << location);
			}
		}

		static bool UnpackFromBits(uint8_t source, int index)
		{
			return (source >> index) & 1;
		}
	};
}
