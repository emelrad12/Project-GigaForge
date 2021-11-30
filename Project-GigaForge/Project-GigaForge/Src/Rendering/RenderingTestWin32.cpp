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
#include <DiligentCore/Graphics/GraphicsEngine/interface/DeviceContext.h>
#include <random>
#include <winerror.h>
#include "TexturedMesh.h"

// import TexturedMesh;
using namespace Diligent;

auto m_GridSize = 50;

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
	RefCntAutoPtr<IPipelineState> texturedMeshPipelineState;
	RefCntAutoPtr<IBuffer> m_CubeVertexBuffer;
	RefCntAutoPtr<IBuffer> m_CubeIndexBuffer;
	RefCntAutoPtr<IBuffer> m_VSConstants;
	RefCntAutoPtr<ITextureView> m_TextureSRV;
	RefCntAutoPtr<IShaderResourceBinding> m_SRB;
	RefCntAutoPtr<IBuffer> m_InstanceBuffer;
	int m_GridSize = 32;
	static constexpr int MaxGridSize = 32;
	static constexpr int MaxInstances = MaxGridSize * MaxGridSize * MaxGridSize;
	float4x4 m_ViewProjMatrix;
	float4x4 m_RotationMatrix;

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


	void CreatePipelineState()
	{
		auto texturedMesh = TexturedMesh();

		CreateInstanceBuffer();
		// clang-format off
		// Define vertex shader input layout
		// This tutorial uses two types of input: per-vertex data and per-instance data.
		LayoutElement LayoutElems[] =
		{
			// Per-vertex data - first buffer slot
			// Attribute 0 - vertex position
			LayoutElement{0, 0, 3, VT_FLOAT32, False},
			// Attribute 1 - texture coordinates
			LayoutElement{1, 0, 2, VT_FLOAT32, False},

			// Per-instance data - second buffer slot
			// We will use four attributes to encode instance-specific 4x4 transformation matrix
			// Attribute 2 - first row
			LayoutElement{2, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
			// Attribute 3 - second row
			LayoutElement{3, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
			// Attribute 4 - third row
			LayoutElement{4, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
			// Attribute 5 - fourth row
			LayoutElement{5, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE}
		};
		// clang-format on

		// Create a shader source stream factory to load shaders from files.
		RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
		m_pEngineFactory->CreateDefaultShaderSourceStreamFactory("Shaders", &pShaderSourceFactory);

		TexturedMesh::CreatePSOInfo CubePsoCI;
		CubePsoCI.pDevice = m_pDevice;
		CubePsoCI.RTVFormat = m_pSwapChain->GetDesc().ColorBufferFormat;
		CubePsoCI.DSVFormat = m_pSwapChain->GetDesc().DepthBufferFormat;
		CubePsoCI.pShaderSourceFactory = pShaderSourceFactory;
		CubePsoCI.VSFilePath = "Vertex.hlsl";
		CubePsoCI.PSFilePath = "Pixel.hlsl";
		CubePsoCI.ExtraLayoutElements = LayoutElems;
		CubePsoCI.NumExtraLayoutElements = _countof(LayoutElems);

		texturedMeshPipelineState = texturedMesh.CreatePipelineState(CubePsoCI);

		// Create dynamic uniform buffer that will store our transformation matrix
		// Dynamic buffers can be frequently updated by the CPU
		CreateUniformBuffer(m_pDevice, sizeof(float4x4) * 2, "VS constants CB", &m_VSConstants);

		// Since we did not explcitly specify the type for 'Constants' variable, default
		// type (SHADER_RESOURCE_VARIABLE_TYPE_STATIC) will be used. Static variables
		// never change and are bound directly to the pipeline state object.
		texturedMeshPipelineState->GetStaticVariableByName(SHADER_TYPE_VERTEX, "Constants")->Set(m_VSConstants);

		// Since we are using mutable variable, we must create a shader resource binding object
		// http://diligentgraphics.com/2016/03/23/resource-binding-model-in-diligent-engine-2-0/
		texturedMeshPipelineState->CreateShaderResourceBinding(&m_SRB, true);

		m_CubeVertexBuffer = texturedMesh.CreateVertexBuffer(m_pDevice, TexturedMesh::VERTEX_COMPONENT_FLAG_POS_UV);
		m_CubeIndexBuffer = texturedMesh.CreateIndexBuffer(m_pDevice);
		m_TextureSRV = texturedMesh.LoadTexture(m_pDevice, "DGLogo.png")->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
		// Set cube texture SRV in the SRB
		m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Texture")->Set(m_TextureSRV);
	}

	void CreateInstanceBuffer()
	{
		// Create instance data buffer that will store transformation matrices
		BufferDesc InstBuffDesc;
		InstBuffDesc.Name = "Instance data buffer";
		// Use default usage as this buffer will only be updated when grid size changes
		InstBuffDesc.Usage = USAGE_DEFAULT;
		InstBuffDesc.BindFlags = BIND_VERTEX_BUFFER;
		InstBuffDesc.uiSizeInBytes = sizeof(float4x4) * MaxInstances;
		m_pDevice->CreateBuffer(InstBuffDesc, nullptr, &m_InstanceBuffer);
		PopulateInstanceBuffer();
	}

	void PopulateInstanceBuffer()
	{
		// Populate instance data buffer
		std::vector<float4x4> InstanceData(pow(m_GridSize, 3));

		float fGridSize = static_cast<float>(m_GridSize);

		std::mt19937 gen; // Standard mersenne_twister_engine. Use default seed
		// to generate consistent distribution.

		std::uniform_real_distribution scale_distr(0.3f, 1.0f);
		std::uniform_real_distribution offset_distr(-0.15f, +0.15f);
		std::uniform_real_distribution rot_distr(-PI_F, +PI_F);

		float BaseScale = 0.6f / fGridSize;
		int instId = 0;
		for (int x = 0; x < m_GridSize; ++x)
		{
			for (int y = 0; y < m_GridSize; ++y)
			{
				for (int z = 0; z < m_GridSize; ++z)
				{
					// Add random offset from central position in the grid
					float xOffset = 2.f * (x + 0.5f + offset_distr(gen)) / fGridSize - 1.f;
					float yOffset = 2.f * (y + 0.5f + offset_distr(gen)) / fGridSize - 1.f;
					float zOffset = 2.f * (z + 0.5f + offset_distr(gen)) / fGridSize - 1.f;
					// Random scale
					float scale = BaseScale * scale_distr(gen);
					// Random rotation+

					float4x4 rotation = float4x4::RotationX(rot_distr(gen)) * float4x4::RotationY(rot_distr(gen)) * float4x4::RotationZ(rot_distr(gen));
					// Combine rotation, scale and translation
					float4x4 matrix = rotation * float4x4::Scale(scale, scale, scale) * float4x4::Translation(xOffset, yOffset, zOffset);
					InstanceData[instId++] = matrix;
				}
			}
		}
		// Update instance data buffer
		Uint32 DataSize = static_cast<Uint32>(sizeof(InstanceData[0]) * InstanceData.size());
		m_pImmediateContext->UpdateBuffer(m_InstanceBuffer, 0, DataSize, InstanceData.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
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
			CBConstants[0] = m_ViewProjMatrix.Transpose();
			CBConstants[1] = m_RotationMatrix.Transpose();
		}

		Uint32 offsets[] = {0, 0};
		IBuffer* pBuffs[] = {m_CubeVertexBuffer, m_InstanceBuffer};
		m_pImmediateContext->SetVertexBuffers(0, _countof(pBuffs), pBuffs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
		m_pImmediateContext->SetIndexBuffer(m_CubeIndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		// Set the pipeline state
		m_pImmediateContext->SetPipelineState(texturedMeshPipelineState);
		// Commit shader resources. RESOURCE_STATE_TRANSITION_MODE_TRANSITION mode
		// makes sure that resources are transitioned to required states.
		m_pImmediateContext->CommitShaderResources(m_SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		DrawIndexedAttribs DrawAttrs; // This is an indexed draw call
		DrawAttrs.IndexType = VT_UINT32; // Index type
		DrawAttrs.NumIndices = 36;
		DrawAttrs.NumInstances = m_GridSize * m_GridSize * m_GridSize; // The number of instances
		// Verify the state of vertex and index buffers
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

		// Camera is at (0, 0, -5) looking along the Z axis
		float4x4 View = float4x4::Translation(0.f, 0.0f, 4.0f);

		// Get pretransform matrix that rotates the scene according the surface orientation
		auto SrfPreTransform = GetSurfacePretransformMatrix(float3{0, 0, 0});

		// Get projection matrix adjusted to the current screen orientation
		auto Proj = GetAdjustedProjectionMatrix(PI_F / 4.0f, 0.1f, 100.f);

		// Compute world-view-projection matrix
		m_ViewProjMatrix = View * SrfPreTransform * Proj;

		// Global rotation matrix
		m_RotationMatrix = float4x4::RotationY(CurrTime * 1.0f) * float4x4::RotationX(-CurrTime * 0.25f);
		PopulateInstanceBuffer();
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

std::unique_ptr<Tutorial00App> app;

LRESULT CALLBACK MessageProc(HWND, UINT, WPARAM, LPARAM);
// Main
int WINAPI RenderInit(HINSTANCE instance, int cmdShow)
{
#if defined(_DEBUG) || defined(DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	app.reset(new Tutorial00App);

	const auto* cmdLine = GetCommandLineA();
	if (!app->ProcessCommandLine(cmdLine))
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

	if (!app->InitializeDiligentEngine(wnd))
		return -1;

	app->CreatePipelineState();


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
			app->Update();
			app->Render();
			app->Present();
		}
	}

	app.reset();

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
		if (app)
		{
			app->WindowResize(LOWORD(lParam), HIWORD(lParam));
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
