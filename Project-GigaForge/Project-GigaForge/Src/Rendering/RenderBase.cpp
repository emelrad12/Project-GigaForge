#include "RenderBase.h"

bool RenderBase::InitializeDiligentEngine(HWND hWnd)
{
	SwapChainDesc SCDesc;
	switch (m_DeviceType)
	{
	case RENDER_DEVICE_TYPE_D3D12:
	{
		EngineD3D12CreateInfo EngineCI;

		m_pEngineFactory = GetEngineFactoryD3D12();
		m_pEngineFactory->CreateDeviceAndContextsD3D12(EngineCI, &m_pDevice, &m_pImmediateContext);
		Win32NativeWindow Window{ hWnd };
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
