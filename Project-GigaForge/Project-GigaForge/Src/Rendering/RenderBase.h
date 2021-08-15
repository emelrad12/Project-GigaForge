#pragma once
#include "RenderingGlobals.h"
#include <activation.h>
#include <BasicMath.hpp>
#include <EngineFactoryD3D12.h>
#include <RefCntAutoPtr.hpp>
using namespace Diligent;
class RenderBase
{
public:
	RenderBase();
	~RenderBase();
	RefCntAutoPtr<IEngineFactoryD3D12> m_pEngineFactory;
	RefCntAutoPtr<IPipelineState> m_pPSO;
	RefCntAutoPtr<IShaderResourceBinding> m_pSRB;
	RefCntAutoPtr<IBuffer> m_CubeVertexBuffer;
	RefCntAutoPtr<IBuffer> m_CubeIndexBuffer;
	RefCntAutoPtr<IBuffer> m_VSConstants;
	float4x4 m_WorldViewProjMatrix;
	bool InitializeDiligentEngine(HWND hWnd);
private:
	RefCntAutoPtr<IRenderDevice> m_pDevice;
	RefCntAutoPtr<IDeviceContext> m_pImmediateContext;
	RefCntAutoPtr<ISwapChain> m_pSwapChain;
	RENDER_DEVICE_TYPE m_DeviceType = RENDER_DEVICE_TYPE_D3D11;
};
