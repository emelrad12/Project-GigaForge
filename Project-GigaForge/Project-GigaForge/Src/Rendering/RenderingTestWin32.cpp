#pragma once
#include "RenderingGlobals.h"
#include <memory>
#include <iomanip>
#include <iostream>

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <Windows.h>
#include <crtdbg.h>
#include "RenderingTestWin32.h"
#include "EngineFactoryD3D12.h"
#include "RenderDevice.h"
#include "DeviceContext.h"
#include "SwapChain.h"

#include "RefCntAutoPtr.hpp"
#include <DiligentCore/Common/interface/BasicMath.hpp>
#include <DiligentCore/Graphics/GraphicsTools/interface/MapHelper.hpp>
#include <DiligentTools/TextureLoader/interface/TextureLoader.h>
#include <DiligentTools/TextureLoader/interface/TextureUtilities.h>
#include <DiligentCore/Graphics/GraphicsTools/interface/GraphicsUtilities.h>
using namespace Diligent;

struct Vertex
{
	float3 pos;
	float2 uv;
};


class Tutorial00App
{
public:
	Tutorial00App()
	{
	}

	~Tutorial00App()
	{
		m_pImmediateContext->Flush();
	}

	RefCntAutoPtr<IEngineFactoryD3D12> m_pEngineFactory;
	RefCntAutoPtr<IPipelineState> m_pPSO;
	RefCntAutoPtr<IBuffer> m_CubeVertexBuffer;
	RefCntAutoPtr<IBuffer> m_CubeIndexBuffer;
	RefCntAutoPtr<IBuffer> m_VSConstants;
	float4x4 m_WorldViewProjMatrix;
	RefCntAutoPtr<ITextureView> m_TextureSRV;
	RefCntAutoPtr<IShaderResourceBinding> m_SRB;

	bool InitializeDiligentEngine(HWND hWnd)
	{
		SwapChainDesc SCDesc;
		switch (m_DeviceType)
		{
		case RENDER_DEVICE_TYPE_D3D12:
			{
				EngineD3D12CreateInfo EngineCI;

				m_pEngineFactory = GetEngineFactoryD3D12();
				m_pEngineFactory->CreateDeviceAndContextsD3D12(EngineCI, &m_pDevice, &m_pImmediateContext);
				Win32NativeWindow Window{hWnd};
				m_pEngineFactory->CreateSwapChainD3D12(m_pDevice, m_pImmediateContext, SCDesc, FullScreenModeDesc{}, Window, &m_pSwapChain);
			}
			break;
		default:
			std::cerr << "Unknown/unsupported device type";
			return false;
			break;
		}

		return true;
	}

	bool ProcessCommandLine(const char* CmdLine)
	{
		const auto* Key = "-mode ";
		const auto* pos = strstr(CmdLine, Key);
		if (pos != nullptr)
		{
			pos += strlen(Key);
			if (_stricmp(pos, "D3D12") == 0)
			{
				m_DeviceType = RENDER_DEVICE_TYPE_D3D12;
			}
			else
			{
				std::cerr << "Unknown device type. Only the following types are supported: D3D12";
				return false;
			}
		}
		else
		{
			m_DeviceType = RENDER_DEVICE_TYPE_D3D12;
		}
		return true;
	}


	void CreateResources()
	{
		// Pipeline state object encompasses configuration of all GPU stages

		GraphicsPipelineStateCreateInfo PSOCreateInfo;

		// Pipeline state name is used by the engine to report issues.
		// It is always a good idea to give objects descriptive names.
		PSOCreateInfo.PSODesc.Name = "Cube PSO";

		// This is a graphics pipeline
		PSOCreateInfo.PSODesc.PipelineType = PIPELINE_TYPE_GRAPHICS;

		// clang-format off
		// This tutorial will render to a single render target
		PSOCreateInfo.GraphicsPipeline.NumRenderTargets = 1;
		// Set render target format which is the format of the swap chain's color buffer
		PSOCreateInfo.GraphicsPipeline.RTVFormats[0] = m_pSwapChain->GetDesc().ColorBufferFormat;
		// Use the depth buffer format from the swap chain
		PSOCreateInfo.GraphicsPipeline.DSVFormat = m_pSwapChain->GetDesc().DepthBufferFormat;
		// Primitive topology defines what kind of primitives will be rendered by this pipeline state
		PSOCreateInfo.GraphicsPipeline.PrimitiveTopology = PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		PSOCreateInfo.GraphicsPipeline.RasterizerDesc.CullMode = CULL_MODE_BACK;

		PSOCreateInfo.GraphicsPipeline.DepthStencilDesc.DepthEnable = true;


		ShaderCreateInfo ShaderCI;
		ShaderCI.SourceLanguage = SHADER_SOURCE_LANGUAGE_HLSL;
		// OpenGL backend requires emulated combined HLSL texture samplers (g_Texture + g_Texture_sampler combination)
		ShaderCI.UseCombinedTextureSamplers = true;
		RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
		const auto* SearchDirectories = R"(C:\Users\emelrad12\Documents\GitHub\Project-GigaForge\Project-GigaForge\Project-GigaForge\Src\Rendering\Shaders\)";
		m_pEngineFactory->CreateDefaultShaderSourceStreamFactory(SearchDirectories, &pShaderSourceFactory);
		ShaderCI.pShaderSourceStreamFactory = pShaderSourceFactory;

		RefCntAutoPtr<IShader> pVS;
		{
			ShaderCI.Desc.ShaderType = SHADER_TYPE_VERTEX;
			ShaderCI.EntryPoint = "main";
			ShaderCI.Desc.Name = "Cube VS";
			ShaderCI.FilePath = "Vertex.hlsl";
			m_pDevice->CreateShader(ShaderCI, &pVS);
			CreateUniformBuffer(m_pDevice, sizeof(float4x4), "VS constants CB", &m_VSConstants);
		}

		// Create a pixel shader
		RefCntAutoPtr<IShader> pPS;
		{
			ShaderCI.Desc.ShaderType = SHADER_TYPE_PIXEL;
			ShaderCI.EntryPoint = "main";
			ShaderCI.Desc.Name = "Cube PS";
			ShaderCI.FilePath = "Pixel.hlsl";
			m_pDevice->CreateShader(ShaderCI, &pPS);
		}

		LayoutElement LayoutElems[] =
		{
			// Attribute 0 - vertex position
			LayoutElement{0, 0, 3, VT_FLOAT32, False},
			// Attribute 1 - vertex color
			LayoutElement{1, 0, 2, VT_FLOAT32, False}
		};


		// Finally, create the pipeline state
		PSOCreateInfo.pVS = pVS;
		PSOCreateInfo.pPS = pPS;

		PSOCreateInfo.GraphicsPipeline.InputLayout.LayoutElements = LayoutElems;
		PSOCreateInfo.GraphicsPipeline.InputLayout.NumElements = _countof(LayoutElems);
		PSOCreateInfo.PSODesc.ResourceLayout.DefaultVariableType = SHADER_RESOURCE_VARIABLE_TYPE_STATIC;

		ShaderResourceVariableDesc Vars[] =
		{
			{SHADER_TYPE_PIXEL, "g_Texture", SHADER_RESOURCE_VARIABLE_TYPE_MUTABLE}
		};

		PSOCreateInfo.PSODesc.ResourceLayout.Variables = Vars;
		PSOCreateInfo.PSODesc.ResourceLayout.NumVariables = _countof(Vars);

		SamplerDesc SamLinearClampDesc
		{
			FILTER_TYPE_LINEAR, FILTER_TYPE_LINEAR, FILTER_TYPE_LINEAR,
			TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP, TEXTURE_ADDRESS_CLAMP
		};

		ImmutableSamplerDesc ImtblSamplers[] =
		{
			{SHADER_TYPE_PIXEL, "g_Texture", SamLinearClampDesc}
		};

		PSOCreateInfo.PSODesc.ResourceLayout.ImmutableSamplers = ImtblSamplers;
		PSOCreateInfo.PSODesc.ResourceLayout.NumImmutableSamplers = _countof(ImtblSamplers);


		m_pDevice->CreateGraphicsPipelineState(PSOCreateInfo, &m_pPSO);
		m_pPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "Constants")->Set(m_VSConstants);
		m_pPSO->CreateShaderResourceBinding(&m_SRB, true);
	}

	void LoadTexture()
	{
		TextureLoadInfo loadInfo;
		loadInfo.IsSRGB = true;
		RefCntAutoPtr<ITexture> Tex;
		CreateTextureFromFile("DGLogo.png", loadInfo, m_pDevice, &Tex);
		m_TextureSRV = Tex->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Texture")->Set(m_TextureSRV);
	}

	void CreateVertexBuffer()
	{
		// Cube vertices

		//      (-1,+1,+1)________________(+1,+1,+1)
		//               /|              /|
		//              / |             / |
		//             /  |            /  |
		//            /   |           /   |
		//(-1,-1,+1) /____|__________/(+1,-1,+1)
		//           |    |__________|____|
		//           |   /(-1,+1,-1) |    /(+1,+1,-1)
		//           |  /            |   /
		//           | /             |  /
		//           |/              | /
		//           /_______________|/
		//        (-1,-1,-1)       (+1,-1,-1)
		//
		
		Vertex CubeVerts[] =
		{
			{float3(-1,-1,-1), float2(0,1)},
			{float3(-1,+1,-1), float2(0,0)},
			{float3(+1,+1,-1), float2(1,0)},
			{float3(+1,-1,-1), float2(1,1)},
			
			{float3(-1,-1,-1), float2(0,1)},
			{float3(-1,-1,+1), float2(0,0)},
			{float3(+1,-1,+1), float2(1,0)},
			{float3(+1,-1,-1), float2(1,1)},
			
			{float3(+1,-1,-1), float2(0,1)},
			{float3(+1,-1,+1), float2(1,1)},
			{float3(+1,+1,+1), float2(1,0)},
			{float3(+1,+1,-1), float2(0,0)},
			
			{float3(+1,+1,-1), float2(0,1)},
			{float3(+1,+1,+1), float2(0,0)},
			{float3(-1,+1,+1), float2(1,0)},
			{float3(-1,+1,-1), float2(1,1)},
			
			{float3(-1,+1,-1), float2(1,0)},
			{float3(-1,+1,+1), float2(0,0)},
			{float3(-1,-1,+1), float2(0,1)},
			{float3(-1,-1,-1), float2(1,1)},
			
			{float3(-1,-1,+1), float2(1,1)},
			{float3(+1,-1,+1), float2(0,1)},
			{float3(+1,+1,+1), float2(0,0)},
			{float3(-1,+1,+1), float2(1,0)}
		};


		BufferDesc VertBuffDesc;
		VertBuffDesc.Name = "Cube vertex buffer";
		VertBuffDesc.Usage = USAGE_IMMUTABLE;
		VertBuffDesc.BindFlags = BIND_VERTEX_BUFFER;
		VertBuffDesc.uiSizeInBytes = sizeof(CubeVerts);
		BufferData VBData;
		VBData.pData = CubeVerts;
		VBData.DataSize = sizeof(CubeVerts);
		m_pDevice->CreateBuffer(VertBuffDesc, &VBData, &m_CubeVertexBuffer);
	}

	void CreateIndexBuffer()
	{
		Uint32 Indices[] =
		{
			2, 0, 1, 2, 3, 0,
			4, 6, 5, 4, 7, 6,
			8, 10, 9, 8, 11, 10,
			12, 14, 13, 12, 15, 14,
			16, 18, 17, 16, 19, 18,
			20, 21, 22, 20, 22, 23
		};

		BufferDesc IndBuffDesc;
		IndBuffDesc.Name = "Cube index buffer";
		IndBuffDesc.Usage = USAGE_IMMUTABLE;
		IndBuffDesc.BindFlags = BIND_INDEX_BUFFER;
		IndBuffDesc.uiSizeInBytes = sizeof(Indices);
		BufferData IBData;
		IBData.pData = Indices;
		IBData.DataSize = sizeof(Indices);
		m_pDevice->CreateBuffer(IndBuffDesc, &IBData, &m_CubeIndexBuffer);
	}

	void Render()
	{
		auto* pRTV = m_pSwapChain->GetCurrentBackBufferRTV();
		auto* pDSV = m_pSwapChain->GetDepthBufferDSV();
		m_pImmediateContext->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		const float ClearColor[] = {0, 0, 0, 1.0f};
		m_pImmediateContext->ClearRenderTarget(pRTV, ClearColor, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		m_pImmediateContext->ClearDepthStencil(pDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		{
			// Map the buffer and write current world-view-projection matrix
			MapHelper<float4x4> CBConstants(m_pImmediateContext, m_VSConstants, MAP_WRITE, MAP_FLAG_DISCARD);
			*CBConstants = m_WorldViewProjMatrix.Transpose();
		}

		Uint32 offset = 0;
		IBuffer* pBuffs[] = {m_CubeVertexBuffer};
		m_pImmediateContext->SetVertexBuffers(0, 1, pBuffs, &offset, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		m_pImmediateContext->SetIndexBuffer(m_CubeIndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		
		m_pImmediateContext->SetPipelineState(m_pPSO);
		m_pImmediateContext->CommitShaderResources(m_SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		DrawIndexedAttribs DrawAttrs;
		DrawAttrs.IndexType = VT_UINT32;
		DrawAttrs.NumIndices = 36;

		DrawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
		m_pImmediateContext->DrawIndexed(DrawAttrs);
	}

	void Present()
	{
		m_pSwapChain->Present();
	}

	void WindowResize(Uint32 Width, Uint32 Height)
	{
		if (m_pSwapChain)
			m_pSwapChain->Resize(Width, Height);
	}

	float CurrTime;

	void Update()
	{
		float ElapsedTime = 0.1;
		CurrTime += ElapsedTime;
		// Apply rotation
		float4x4 CubeModelTransform = float4x4::RotationY(static_cast<float>(CurrTime) * 1.0f) * float4x4::RotationX(-PI_F * 0.1f);

		// Camera is at (0, 0, -5) looking along the Z axis
		float4x4 View = float4x4::Translation(0.f, 0.0f, 5.0f);

		// Get pretransform matrix that rotates the scene according the surface orientation
		auto SrfPreTransform = GetSurfacePretransformMatrix(float3{0, 0, 0});

		// Get projection matrix adjusted to the current screen orientation
		auto Proj = GetAdjustedProjectionMatrix(PI_F / 4.0f, 0.1f, 100.f);

		// Compute world-view-projection matrix
		m_WorldViewProjMatrix = CubeModelTransform * View * SrfPreTransform * Proj;
	}

	float4x4 GetSurfacePretransformMatrix(const float3& f3CameraViewAxis) const
	{
		const auto& SCDesc = m_pSwapChain->GetDesc();
		switch (SCDesc.PreTransform)
		{
		case SURFACE_TRANSFORM_ROTATE_90:
			// The image content is rotated 90 degrees clockwise.
			return float4x4::RotationArbitrary(f3CameraViewAxis, -PI_F / 2.f);

		case SURFACE_TRANSFORM_ROTATE_180:
			// The image content is rotated 180 degrees clockwise.
			return float4x4::RotationArbitrary(f3CameraViewAxis, -PI_F);

		case SURFACE_TRANSFORM_ROTATE_270:
			// The image content is rotated 270 degrees clockwise.
			return float4x4::RotationArbitrary(f3CameraViewAxis, -PI_F * 3.f / 2.f);

		case SURFACE_TRANSFORM_OPTIMAL:
			UNEXPECTED("SURFACE_TRANSFORM_OPTIMAL is only valid as parameter during swap chain initialization.");
			return float4x4::Identity();

		case SURFACE_TRANSFORM_HORIZONTAL_MIRROR:
		case SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90:
		case SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180:
		case SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270:
			UNEXPECTED("Mirror transforms are not supported");
			return float4x4::Identity();

		default:
			return float4x4::Identity();
		}
	}

	float4x4 GetAdjustedProjectionMatrix(float FOV, float NearPlane, float FarPlane) const
	{
		const auto& SCDesc = m_pSwapChain->GetDesc();

		float AspectRatio = static_cast<float>(SCDesc.Width) / static_cast<float>(SCDesc.Height);
		float XScale, YScale;
		if (SCDesc.PreTransform == SURFACE_TRANSFORM_ROTATE_90 ||
			SCDesc.PreTransform == SURFACE_TRANSFORM_ROTATE_270 ||
			SCDesc.PreTransform == SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90 ||
			SCDesc.PreTransform == SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270)
		{
			// When the screen is rotated, vertical FOV becomes horizontal FOV
			XScale = 1.f / std::tan(FOV / 2.f);
			// Aspect ratio is inversed
			YScale = XScale * AspectRatio;
		}
		else
		{
			YScale = 1.f / std::tan(FOV / 2.f);
			XScale = YScale / AspectRatio;
		}

		float4x4 Proj;
		Proj._11 = XScale;
		Proj._22 = YScale;
		Proj.SetNearFarClipPlanes(NearPlane, FarPlane, m_pDevice->GetDeviceInfo().IsGLDevice());
		return Proj;
	}

	RENDER_DEVICE_TYPE GetDeviceType() const { return m_DeviceType; }

private:
	RefCntAutoPtr<IRenderDevice> m_pDevice;
	RefCntAutoPtr<IDeviceContext> m_pImmediateContext;
	RefCntAutoPtr<ISwapChain> m_pSwapChain;
	RENDER_DEVICE_TYPE m_DeviceType = RENDER_DEVICE_TYPE_D3D11;
};

std::unique_ptr<Tutorial00App> g_pTheApp;

LRESULT CALLBACK MessageProc(HWND, UINT, WPARAM, LPARAM);
// Main
int WINAPI RenderInit(HINSTANCE instance, int cmdShow)
{
#if defined(_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	g_pTheApp.reset(new Tutorial00App);

	const auto* cmdLine = GetCommandLineA();
	if (!g_pTheApp->ProcessCommandLine(cmdLine))
		return -1;

	std::wstring Title(L"Tutorial00: Hello Win32");
	Title.append(L" (D3D12)");
	// Register our window class
	WNDCLASSEX wcex = {
		sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, MessageProc,
		0L, 0L, instance, NULL, NULL, NULL, NULL, L"SampleApp", NULL
	};
	RegisterClassEx(&wcex);

	// Create a window
	LONG WindowWidth = 1280;
	LONG WindowHeight = 1024;
	RECT rc = {0, 0, WindowWidth, WindowHeight};
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
	auto wnd = CreateWindow(L"SampleApp", Title.c_str(),
	                        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
	                        rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, instance, NULL);
	if (!wnd)
	{
		MessageBox(NULL, L"Cannot create window", L"Error", MB_OK | MB_ICONERROR);
		return 0;
	}
	ShowWindow(wnd, cmdShow);
	UpdateWindow(wnd);

	if (!g_pTheApp->InitializeDiligentEngine(wnd))
		return -1;

	g_pTheApp->CreateResources();
	g_pTheApp->CreateVertexBuffer();
	g_pTheApp->CreateIndexBuffer();
	g_pTheApp->LoadTexture();

	// Main message loop
	MSG msg = {0};
	while (WM_QUIT != msg.message)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			g_pTheApp->Update();
			g_pTheApp->Render();
			g_pTheApp->Present();
		}
	}

	g_pTheApp.reset();

	return (int)msg.wParam;
}

// Called every time the NativeNativeAppBase receives a message
LRESULT CALLBACK MessageProc(HWND wnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_PAINT:
		{
			PAINTSTRUCT ps;
			BeginPaint(wnd, &ps);
			EndPaint(wnd, &ps);
			return 0;
		}
	case WM_SIZE: // Window size has been changed
		if (g_pTheApp)
		{
			g_pTheApp->WindowResize(LOWORD(lParam), HIWORD(lParam));
		}
		return 0;

	case WM_CHAR:
		if (wParam == VK_ESCAPE)
			PostQuitMessage(0);
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;

	case WM_GETMINMAXINFO:
		{
			LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam;

			lpMMI->ptMinTrackSize.x = 320;
			lpMMI->ptMinTrackSize.y = 240;
			return 0;
		}

	default:
		return DefWindowProc(wnd, message, wParam, lParam);
	}
}
