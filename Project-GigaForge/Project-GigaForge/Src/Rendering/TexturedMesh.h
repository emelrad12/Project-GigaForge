#pragma once
#include "RenderingGlobals.h"

#include <array>

#include "RenderDevice.h"
#include "Buffer.h"
#include "RefCntAutoPtr.hpp"
#include "BasicMath.hpp"

namespace Diligent
{
	class TexturedMesh
	{
	public:
		static constexpr Uint32 NumVertices = 4 * 6;
		static constexpr Uint32 NumIndices = 3 * 2 * 6;

		const std::array<float3, NumVertices> Positions = {
			float3{-1, -1, -1}, float3{-1, +1, -1}, float3{+1, +1, -1}, float3{+1, -1, -1}, // Bottom
			float3{-1, -1, -1}, float3{-1, -1, +1}, float3{+1, -1, +1}, float3{+1, -1, -1}, // Front
			float3{+1, -1, -1}, float3{+1, -1, +1}, float3{+1, +1, +1}, float3{+1, +1, -1}, // Right
			float3{+1, +1, -1}, float3{+1, +1, +1}, float3{-1, +1, +1}, float3{-1, +1, -1}, // Back
			float3{-1, +1, -1}, float3{-1, +1, +1}, float3{-1, -1, +1}, float3{-1, -1, -1}, // Left
			float3{-1, -1, +1}, float3{+1, -1, +1}, float3{+1, +1, +1}, float3{-1, +1, +1} // Top
		};

		const std::array<float2, NumVertices> Texcoords = {
			float2{0, 1}, float2{0, 0}, float2{1, 0}, float2{1, 1}, // Bottom
			float2{0, 1}, float2{0, 0}, float2{1, 0}, float2{1, 1}, // Front
			float2{0, 1}, float2{1, 1}, float2{1, 0}, float2{0, 0}, // Right
			float2{0, 1}, float2{0, 0}, float2{1, 0}, float2{1, 1}, // Back
			float2{1, 0}, float2{0, 0}, float2{0, 1}, float2{1, 1}, // Left
			float2{1, 1}, float2{0, 1}, float2{0, 0}, float2{1, 0} // Top
		};

		const std::array<float3, NumVertices> Normals = {
			float3{0, 0, -1}, float3{0, 0, -1}, float3{0, 0, -1}, float3{0, 0, -1}, // Bottom
			float3{0, -1, 0}, float3{0, -1, 0}, float3{0, -1, 0}, float3{0, -1, 0}, // Front
			float3{+1, 0, 0}, float3{+1, 0, 0}, float3{+1, 0, 0}, float3{+1, 0, 0}, // Right
			float3{0, +1, 0}, float3{0, +1, 0}, float3{0, +1, 0}, float3{0, +1, 0}, // Back
			float3{-1, 0, 0}, float3{-1, 0, 0}, float3{-1, 0, 0}, float3{-1, 0, 0}, // Left
			float3{0, 0, +1}, float3{0, 0, +1}, float3{0, 0, +1}, float3{0, 0, +1} // Top
		};

		const std::array<Uint32, NumIndices> Indices =
		{
			2, 0, 1, 2, 3, 0,
			4, 6, 5, 4, 7, 6,
			8, 10, 9, 8, 11, 10,
			12, 14, 13, 12, 15, 14,
			16, 18, 17, 16, 19, 18,
			20, 21, 22, 20, 22, 23
		};

		enum VERTEX_COMPONENT_FLAGS : Uint32
		{
			VERTEX_COMPONENT_FLAG_NONE = 0x00,
			VERTEX_COMPONENT_FLAG_POSITION = 0x01,
			VERTEX_COMPONENT_FLAG_NORMAL = 0x02,
			VERTEX_COMPONENT_FLAG_TEXCOORD = 0x04,

			VERTEX_COMPONENT_FLAG_POS_UV =
			VERTEX_COMPONENT_FLAG_POSITION |
			VERTEX_COMPONENT_FLAG_TEXCOORD,

			VERTEX_COMPONENT_FLAG_POS_NORM_UV =
			VERTEX_COMPONENT_FLAG_POSITION |
			VERTEX_COMPONENT_FLAG_NORMAL |
			VERTEX_COMPONENT_FLAG_TEXCOORD
		};

		// DEFINE_FLAG_ENUM_OPERATORS(VERTEX_COMPONENT_FLAGS);

		RefCntAutoPtr<IBuffer> CreateVertexBuffer(IRenderDevice* pDevice, VERTEX_COMPONENT_FLAGS Components, BIND_FLAGS BindFlags = BIND_VERTEX_BUFFER, BUFFER_MODE Mode = BUFFER_MODE_UNDEFINED);
		RefCntAutoPtr<IBuffer> CreateIndexBuffer(IRenderDevice* pDevice, BIND_FLAGS BindFlags = BIND_INDEX_BUFFER, BUFFER_MODE Mode = BUFFER_MODE_UNDEFINED);
		RefCntAutoPtr<ITexture> LoadTexture(IRenderDevice* pDevice, const char* Path);

		struct CreatePSOInfo
		{
			IRenderDevice* pDevice = nullptr;
			TEXTURE_FORMAT RTVFormat = TEX_FORMAT_UNKNOWN;
			TEXTURE_FORMAT DSVFormat = TEX_FORMAT_UNKNOWN;
			IShaderSourceInputStreamFactory* pShaderSourceFactory = nullptr;
			const char* VSFilePath = nullptr;
			const char* PSFilePath = nullptr;
			VERTEX_COMPONENT_FLAGS Components = VERTEX_COMPONENT_FLAG_NONE;
			LayoutElement* ExtraLayoutElements = nullptr;
			Uint32 NumExtraLayoutElements = 0;
			Uint8 SampleCount = 1;
		};

		RefCntAutoPtr<IPipelineState> CreatePipelineState(const CreatePSOInfo& CreateInfo);
		
	};
}
