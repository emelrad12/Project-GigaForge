#include <vector>

#include "TexturedMesh.h"
#include "BasicMath.hpp"
#include <DiligentTools/TextureLoader/interface/TextureUtilities.h>

namespace Diligent
{
	//      (-1,+1,+1)________________(+1,+1,+1)                  Z
	//               /|              /|                           |      Y
	//              / |             / |                           |     /
	//             /  |            /  |                           |    /
	//            /   |           /   |                           |   /
	//(-1,-1,+1) /____|__________/(+1,-1,+1)                      |  /
	//           |    |__________|____|                           | /
	//           |   /(-1,+1,-1) |    /(+1,+1,-1)                 |----------------> X
	//           |  /            |   /
	//           | /             |  /
	//           |/              | /
	//           /_______________|/
	//        (-1,-1,-1)       (+1,-1,-1)
	//

	RefCntAutoPtr<IBuffer> TexturedMesh::CreateVertexBuffer(IRenderDevice* pDevice, VERTEX_COMPONENT_FLAGS Components, BIND_FLAGS BindFlags, BUFFER_MODE Mode)
	{
		VERIFY_EXPR(Components != VERTEX_COMPONENT_FLAG_NONE);
		const Uint32 TotalVertexComponents =
			((Components & VERTEX_COMPONENT_FLAG_POSITION) ? 3 : 0) +
			((Components & VERTEX_COMPONENT_FLAG_NORMAL) ? 3 : 0) +
			((Components & VERTEX_COMPONENT_FLAG_TEXCOORD) ? 2 : 0);

		std::vector<float> VertexData(TotalVertexComponents * NumVertices);

		auto it = VertexData.begin();
		for (Uint32 v = 0; v < NumVertices; ++v)
		{
			if (Components & VERTEX_COMPONENT_FLAG_POSITION)
			{
				const auto& Pos{Positions[v]};
				*it++ = Pos.x;
				*it++ = Pos.y;
				*it++ = Pos.z;
			}
			if (Components & VERTEX_COMPONENT_FLAG_NORMAL)
			{
				const auto& N{Normals[v]};
				*it++ = N.x;
				*it++ = N.y;
				*it++ = N.z;
			}
			if (Components & VERTEX_COMPONENT_FLAG_TEXCOORD)
			{
				const auto& UV{Texcoords[v]};
				*it++ = UV.x;
				*it++ = UV.y;
			}
		}
		VERIFY_EXPR(it == VertexData.end());

		BufferDesc VertBuffDesc;
		VertBuffDesc.Name = "Cube vertex buffer";
		VertBuffDesc.Usage = USAGE_IMMUTABLE;
		VertBuffDesc.BindFlags = BindFlags;
		VertBuffDesc.uiSizeInBytes = static_cast<Uint32>(VertexData.size() * sizeof(VertexData[0]));
		VertBuffDesc.Mode = Mode;
		if (Mode != BUFFER_MODE_UNDEFINED)
		{
			VertBuffDesc.ElementByteStride = TotalVertexComponents * sizeof(VertexData[0]);
		}

		BufferData VBData;
		VBData.pData = VertexData.data();
		VBData.DataSize = VertBuffDesc.uiSizeInBytes;
		RefCntAutoPtr<IBuffer> pCubeVertexBuffer;

		pDevice->CreateBuffer(VertBuffDesc, &VBData, &pCubeVertexBuffer);

		return pCubeVertexBuffer;
	}

	RefCntAutoPtr<IBuffer> TexturedMesh::CreateIndexBuffer(IRenderDevice* pDevice, BIND_FLAGS BindFlags, BUFFER_MODE Mode)
	{
		BufferDesc IndBuffDesc;
		IndBuffDesc.Name = "Cube index buffer";
		IndBuffDesc.Usage = USAGE_IMMUTABLE;
		IndBuffDesc.BindFlags = BindFlags;
		IndBuffDesc.uiSizeInBytes = sizeof(Indices);
		IndBuffDesc.Mode = Mode;
		if (Mode != BUFFER_MODE_UNDEFINED)
			IndBuffDesc.ElementByteStride = sizeof(Indices[0]);
		BufferData IBData;
		IBData.pData = Indices.data();
		IBData.DataSize = NumIndices * sizeof(Indices[0]);
		RefCntAutoPtr<IBuffer> pBuffer;
		pDevice->CreateBuffer(IndBuffDesc, &IBData, &pBuffer);
		return pBuffer;
	}

	RefCntAutoPtr<ITexture> TexturedMesh::LoadTexture(IRenderDevice* pDevice, const char* Path)
	{
		TextureLoadInfo loadInfo;
		loadInfo.IsSRGB = true;
		RefCntAutoPtr<ITexture> pTex;
		CreateTextureFromFile(Path, loadInfo, pDevice, &pTex);
		return pTex;
	}

	RefCntAutoPtr<IPipelineState> TexturedMesh::CreatePipelineState(const CreatePSOInfo& CreateInfo)
	{
		GraphicsPipelineStateCreateInfo PSOCreateInfo;
		PipelineStateDesc& PSODesc = PSOCreateInfo.PSODesc;
		PipelineResourceLayoutDesc& ResourceLayout = PSODesc.ResourceLayout;
		GraphicsPipelineDesc& GraphicsPipeline = PSOCreateInfo.GraphicsPipeline;

		// This is a graphics pipeline
		PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

		// Pipeline state name is used by the engine to report issues.
		// It is always a good idea to give objects descriptive names.
		PSODesc.Name = "Cube PSO";

		// clang-format off
		// This tutorial will render to a single render target
		GraphicsPipeline.NumRenderTargets = 1;
		// Set render target format which is the format of the swap chain's color buffer
		GraphicsPipeline.RTVFormats[0] = CreateInfo.RTVFormat;
		// Set depth buffer format which is the format of the swap chain's back buffer
		GraphicsPipeline.DSVFormat = CreateInfo.DSVFormat;
		// Set the desired number of samples
		GraphicsPipeline.SmplDesc.Count = CreateInfo.SampleCount;
		// Primitive topology defines what kind of primitives will be rendered by this pipeline state
		GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		// Cull back faces
		GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_BACK;
		// Enable depth testing
		GraphicsPipeline.DepthStencilDesc.DepthEnable = True;
		// clang-format on
		ShaderCreateInfo ShaderCI;
		// Tell the system that the shader source code is in HLSL.
		// For OpenGL, the engine will convert this into GLSL under the hood.
		ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;

		// OpenGL backend requires emulated combined HLSL texture samplers (g_Texture + g_Texture_sampler combination)
		ShaderCI.UseCombinedTextureSamplers = true;

		ShaderCI.pShaderSourceStreamFactory = CreateInfo.pShaderSourceFactory;
		// Create a vertex shader
		RefCntAutoPtr<IShader> pVS;
		{
			ShaderCI.Desc.ShaderType = SHADER_TYPE_VERTEX;
			ShaderCI.EntryPoint = "main";
			ShaderCI.Desc.Name = "Cube VS";
			ShaderCI.FilePath = CreateInfo.VSFilePath;
			CreateInfo.pDevice->CreateShader(ShaderCI, &pVS);
		}

		// Create a pixel shader
		RefCntAutoPtr<IShader> pPS;
		{
			ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
			ShaderCI.EntryPoint = "main";
			ShaderCI.Desc.Name = "Cube PS";
			ShaderCI.FilePath = CreateInfo.PSFilePath;
			CreateInfo.pDevice->CreateShader(ShaderCI, &pPS);
		}

		std::vector<LayoutElement> LayoutElems;

		Uint32 Attrib = 0;
		if (CreateInfo.Components & VERTEX_COMPONENT_FLAG_POSITION)
			LayoutElems.emplace_back(Attrib++, 0, 3, VT_FLOAT32, False);
		if (CreateInfo.Components & VERTEX_COMPONENT_FLAG_NORMAL)
			LayoutElems.emplace_back(Attrib++, 0, 3, VT_FLOAT32, False);
		if (CreateInfo.Components & VERTEX_COMPONENT_FLAG_TEXCOORD)
			LayoutElems.emplace_back(Attrib++, 0, 2, VT_FLOAT32, False);

		for (Uint32 i = 0; i < CreateInfo.NumExtraLayoutElements; ++i)
			LayoutElems.push_back(CreateInfo.ExtraLayoutElements[i]);

		GraphicsPipeline.InputLayout.LayoutElements = LayoutElems.data();
		GraphicsPipeline.InputLayout.NumElements = static_cast<Uint32>(LayoutElems.size());

		PSOCreateInfo.pVS = pVS;
		PSOCreateInfo.pPS = pPS;

		// Define variable type that will be used by default
		ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

		// Shader variables should typically be mutable, which means they are expected
		// to change on a per-instance basis
		// clang-format off
		ShaderResourceVariableDesc Vars[] =
		{
			{SHADER_TYPE_PIXEL, "g_Texture", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}
		};
		// clang-format on
		ResourceLayout.Variables = Vars;
		ResourceLayout.NumVariables = _countof(Vars);

		// Define immutable sampler for g_Texture. Immutable samplers should be used whenever possible
		// clang-format off
		SamplerDesc SamLinearClampDesc
		{
			FILTER_TYPE_LINEAR, FILTER_TYPE_LINEAR, FILTER_TYPE_LINEAR,
			TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP
		};
		ImmutableSamplerDesc ImtblSamplers[] =
		{
			{SHADER_TYPE_PIXEL, "g_Texture", SamLinearClampDesc}
		};
		// clang-format on
		ResourceLayout.ImmutableSamplers = ImtblSamplers;
		ResourceLayout.NumImmutableSamplers = _countof(ImtblSamplers);

		RefCntAutoPtr<IPipelineState> pPSO;
		CreateInfo.pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &pPSO);
		return pPSO;
	}
}
