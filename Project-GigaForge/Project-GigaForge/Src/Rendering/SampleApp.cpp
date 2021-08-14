#include "RenderingGlobals.h"
#include <sstream>

#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include "SampleApp.h"
#include <activation.h>

#include "DiligentCore/Primitives/interface/Errors.hpp"
#include "StringTools.hpp"
#include "DiligentTools/TextureLoader/interface/Image.h"
#include "FileWrapper.hpp"
#include "EngineFactoryD3D12.h"

#include "DiligentTools/ThirdParty/imgui/imgui.h"
#include "DiligentTools/Imgui/interface/ImGuiImplDiligent.hpp"
#include "DiligentTools/Imgui/interface/ImGuiUtils.hpp"

namespace Diligent
{
	SampleApp::SampleApp() :
		m_TheSample{CreateSample()},
		m_AppTitle{m_TheSample->GetSampleName()}
	{
	}

	SampleApp::~SampleApp()
	{
		m_pImGui.reset();
		m_TheSample.reset();

		if (!m_pDeviceContexts.empty())
		{
			for (Uint32 q = 0; q < m_NumImmediateContexts; ++q)
				m_pDeviceContexts[q]->Flush();
			m_pDeviceContexts.clear();
		}
		m_NumImmediateContexts = 0;
		m_pSwapChain.Release();
		m_pDevice.Release();
	}


	void SampleApp::InitializeDiligentEngine(const NativeWindow* pWindow)
	{
		if (m_ScreenCaptureInfo.AllowCapture)
			m_SwapChainInitDesc.Usage |= SWAP_CHAIN_USAGE_COPY_SOURCE;

#if PLATFORM_MACOS
        // We need at least 3 buffers in Metal to avoid massive
        // peformance degradation in full screen mode.
        // https://github.com/KhronosGroup/MoltenVK/issues/808
        m_SwapChainInitDesc.BufferCount = 3;
#endif

		Uint32 NumImmediateContexts = 0;

		std::vector<IDeviceContext*> ppContexts;
		switch (m_DeviceType)
		{
#if D3D12_SUPPORTED
		case RENDER_DEVICE_TYPE_D3D12:
			{
#    if ENGINE_DLL
            // Load the dll and import GetEngineFactoryD3D12() function
            auto GetEngineFactoryD3D12 = LoadGraphicsEngineD3D12();
#    endif
				auto* pFactoryD3D12 = GetEngineFactoryD3D12();
				if (!pFactoryD3D12->LoadD3D12())
				{
					LOG_ERROR_AND_THROW("Failed to load Direct3D12");
				}
				m_pEngineFactory = pFactoryD3D12;

				EngineD3D12CreateInfo EngineCI;
				EngineCI.GraphicsAPIVersion = {11, 0};
				if (m_ValidationLevel >= 0)
					EngineCI.SetValidationLevel(static_cast<VALIDATION_LEVEL>(m_ValidationLevel));

				Uint32 NumAdapters = 0;
				pFactoryD3D12->EnumerateAdapters(EngineCI.GraphicsAPIVersion, NumAdapters, nullptr);
				std::vector<GraphicsAdapterInfo> Adapters(NumAdapters);
				if (NumAdapters > 0)
				{
					pFactoryD3D12->EnumerateAdapters(EngineCI.GraphicsAPIVersion, NumAdapters, Adapters.data());
				}
				else
				{
#    if D3D11_SUPPORTED
                LOG_ERROR_MESSAGE("Failed to find Direct3D12-compatible hardware adapters. Attempting to initialize the engine in Direct3D11 mode.");
                m_DeviceType = RENDER_DEVICE_TYPE_D3D11;
                InitializeDiligentEngine(pWindow);
                return;
#    else
					LOG_ERROR_AND_THROW("Failed to find Direct3D12-compatible hardware adapters.");
#    endif
				}

				EngineCI.AdapterId = m_AdapterId;
				if (m_AdapterType == ADAPTER_TYPE_SOFTWARE)
				{
					for (Uint32 i = 0; i < Adapters.size(); ++i)
					{
						if (Adapters[i].Type == m_AdapterType)
						{
							EngineCI.AdapterId = i;
							LOG_INFO_MESSAGE("Found software adapter '", Adapters[i].Description, "'");
							break;
						}
					}
				}

				m_TheSample->ModifyEngineInitInfo({pFactoryD3D12, m_DeviceType, EngineCI, m_SwapChainInitDesc});

				m_AdapterAttribs = Adapters[EngineCI.AdapterId];
				if (m_AdapterType != ADAPTER_TYPE_SOFTWARE)
				{
					Uint32 NumDisplayModes = 0;
					pFactoryD3D12->EnumerateDisplayModes(EngineCI.GraphicsAPIVersion, EngineCI.AdapterId, 0, TEX_FORMAT_RGBA8_UNORM_SRGB, NumDisplayModes, nullptr);
					m_DisplayModes.resize(NumDisplayModes);
					pFactoryD3D12->EnumerateDisplayModes(EngineCI.GraphicsAPIVersion, EngineCI.AdapterId, 0, TEX_FORMAT_RGBA8_UNORM_SRGB, NumDisplayModes, m_DisplayModes.data());
				}

				NumImmediateContexts = std::max(1u, EngineCI.NumImmediateContexts);
				ppContexts.resize(NumImmediateContexts + EngineCI.NumDeferredContexts);
				pFactoryD3D12->CreateDeviceAndContextsD3D12(EngineCI, &m_pDevice, ppContexts.data());
				if (!m_pDevice)
				{
					LOG_ERROR_AND_THROW("Unable to initialize Diligent Engine in Direct3D12 mode. The API may not be available, "
						"or required features may not be supported by this GPU/driver/OS version.");
				}

				if (!m_pSwapChain && pWindow != nullptr)
					pFactoryD3D12->CreateSwapChainD3D12(m_pDevice, ppContexts[0], m_SwapChainInitDesc, FullScreenModeDesc{}, *pWindow, &m_pSwapChain);
			}
			break;
#endif

		default:
			LOG_ERROR_AND_THROW("Unknown device type");
			break;
		}

		switch (m_DeviceType)
		{
			// clang-format off
		case RENDER_DEVICE_TYPE_D3D11: m_AppTitle.append(" (D3D11");
			break;
		case RENDER_DEVICE_TYPE_D3D12: m_AppTitle.append(" (D3D12");
			break;
		case RENDER_DEVICE_TYPE_GL: m_AppTitle.append(" (OpenGL");
			break;
		case RENDER_DEVICE_TYPE_GLES: m_AppTitle.append(" (OpenGLES");
			break;
		case RENDER_DEVICE_TYPE_VULKAN: m_AppTitle.append(" (Vulkan");
			break;
		case RENDER_DEVICE_TYPE_METAL: m_AppTitle.append(" (Metal");
			break;
		default: UNEXPECTED("Unknown/unsupported device type");
			// clang-format on
		}
		m_AppTitle.append(", API ");
		m_AppTitle.append(std::to_string(DILIGENT_API_VERSION));
		m_AppTitle.push_back(')');

		m_NumImmediateContexts = NumImmediateContexts;
		m_pDeviceContexts.resize(ppContexts.size());
		for (size_t i = 0; i < ppContexts.size(); ++i)
			m_pDeviceContexts[i].Attach(ppContexts[i]);

		if (m_ScreenCaptureInfo.AllowCapture)
		{
			m_pScreenCapture.reset(new ScreenCapture(m_pDevice));
		}
	}

	void SampleApp::InitializeSample()
	{
#if PLATFORM_WIN32
		if (!m_DisplayModes.empty())
		{
			const HWND hDesktop = GetDesktopWindow();

			RECT rc;
			GetWindowRect(hDesktop, &rc);
			Uint32 ScreenWidth = static_cast<Uint32>(rc.right - rc.left);
			Uint32 ScreenHeight = static_cast<Uint32>(rc.bottom - rc.top);
			for (int i = 0; i < static_cast<int>(m_DisplayModes.size()); ++i)
			{
				if (ScreenWidth == m_DisplayModes[i].Width && ScreenHeight == m_DisplayModes[i].Height)
				{
					m_SelectedDisplayMode = i;
					break;
				}
			}
		}
#endif

		const auto& SCDesc = m_pSwapChain->GetDesc();

		m_MaxFrameLatency = SCDesc.BufferCount;

		std::vector<IDeviceContext*> ppContexts(m_pDeviceContexts.size());
		for (size_t ctx = 0; ctx < m_pDeviceContexts.size(); ++ctx)
			ppContexts[ctx] = m_pDeviceContexts[ctx];

		SampleInitInfo InitInfo;
		InitInfo.pEngineFactory = m_pEngineFactory;
		InitInfo.pDevice = m_pDevice;
		InitInfo.ppContexts = ppContexts.data();
		InitInfo.NumImmediateCtx = m_NumImmediateContexts;
		VERIFY_EXPR(m_pDeviceContexts.size() >= m_NumImmediateContexts);
		InitInfo.NumDeferredCtx = static_cast<Uint32>(m_pDeviceContexts.size()) - m_NumImmediateContexts;
		InitInfo.pSwapChain = m_pSwapChain;
		InitInfo.pImGui = m_pImGui.get();
		m_TheSample->Initialize(InitInfo);

		m_TheSample->WindowResize(SCDesc.Width, SCDesc.Height);
	}

	void SampleApp::UpdateAdaptersDialog()
	{
#if PLATFORM_WIN32 || PLATFORM_LINUX
		const auto& SCDesc = m_pSwapChain->GetDesc();

		Uint32 AdaptersWndWidth = std::min(330u, SCDesc.Width);
		ImGui::SetNextWindowSize(ImVec2(static_cast<float>(AdaptersWndWidth), 0), ImGuiCond_Always);
		ImGui::SetNextWindowPos(ImVec2(static_cast<float>(std::max(SCDesc.Width - AdaptersWndWidth, 10U) - 10), 10), ImGuiCond_Always);
		ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Adapters", nullptr, ImGuiWindowFlags_NoResize))
		{
			if (m_AdapterAttribs.Type != ADAPTER_TYPE_UNKNOWN)
			{
				ImGui::TextDisabled("Adapter: %s (%d MB)", m_AdapterAttribs.Description, static_cast<int>(m_AdapterAttribs.Memory.LocalMemory >> 20));
			}

			if (!m_DisplayModes.empty())
			{
				std::vector<const char*> DisplayModes(m_DisplayModes.size());
				std::vector<std::string> DisplayModeStrings(m_DisplayModes.size());
				for (int i = 0; i < static_cast<int>(m_DisplayModes.size()); ++i)
				{
					static constexpr const char* ScalingModeStr[] =
					{
						""
						" Centered",
						" Stretched" //
					};
					const auto& Mode = m_DisplayModes[i];

					std::stringstream ss;

					float RefreshRate = static_cast<float>(Mode.RefreshRateNumerator) / static_cast<float>(Mode.RefreshRateDenominator);
					ss << Mode.Width << "x" << Mode.Height << "@" << std::fixed << std::setprecision(2) << RefreshRate << " Hz" << ScalingModeStr[static_cast<int>(Mode.Scaling)];
					DisplayModeStrings[i] = ss.str();
					DisplayModes[i] = DisplayModeStrings[i].c_str();
				}

				ImGui::SetNextItemWidth(220);
				ImGui::Combo("Display Modes", &m_SelectedDisplayMode, DisplayModes.data(), static_cast<int>(DisplayModes.size()));
			}

			if (m_bFullScreenMode)
			{
				if (ImGui::Button("Go Windowed"))
				{
					SetWindowedMode();
				}
			}
			else
			{
				if (!m_DisplayModes.empty())
				{
					if (ImGui::Button("Go Full Screen"))
					{
						const auto& SelectedMode = m_DisplayModes[m_SelectedDisplayMode];
						SetFullscreenMode(SelectedMode);
					}
				}
			}

			ImGui::Checkbox("VSync", &m_bVSync);

			if (m_pDevice->GetDeviceInfo().IsD3DDevice())
			{
				// clang-format off
				std::pair<Uint32, const char*> FrameLatencies[] =
				{
					{1, "1"},
					{2, "2"},
					{3, "3"},
					{4, "4"},
					{5, "5"},
					{6, "6"},
					{7, "7"},
					{8, "8"},
					{9, "9"},
					{10, "10"}
				};
				// clang-format on

				if (SCDesc.BufferCount <= _countof(FrameLatencies) && m_MaxFrameLatency <= _countof(FrameLatencies))
				{
					ImGui::SetNextItemWidth(120);
					auto NumFrameLatencyItems = std::max(std::max(m_MaxFrameLatency, SCDesc.BufferCount), Uint32{4});
					if (ImGui::Combo("Max frame latency", &m_MaxFrameLatency, FrameLatencies, NumFrameLatencyItems))
					{
						m_pSwapChain->SetMaximumFrameLatency(m_MaxFrameLatency);
					}
				}
				else
				{
					// 10+ buffer swap chain or frame latency? Something is not quite right
				}
			}
		}
		ImGui::End();
#endif
	}


	std::string GetArgument(const char*& pos, const char* ArgName)
	{
		size_t ArgNameLen = 0;
		const char* delimeters = " \n\r";
		while (pos[ArgNameLen] != 0 && strchr(delimeters, pos[ArgNameLen]) == nullptr)
			++ArgNameLen;

		if (StrCmpNoCase(pos, ArgName, ArgNameLen) == 0)
		{
			pos += ArgNameLen;
			while (*pos != 0 && strchr(delimeters, *pos) != nullptr)
				++pos;
			std::string Arg;
			while (*pos != 0 && strchr(delimeters, *pos) == nullptr)
				Arg.push_back(*(pos++));
			return Arg;
		}
		else
		{
			return std::string{};
		}
	}

	// Command line example to capture frames:
	//
	//     -mode d3d11 -adapters_dialog 0 -capture_path . -capture_fps 15 -capture_name frame -width 640 -height 480 -capture_format jpg -capture_quality 100 -capture_frames 3 -capture_alpha 0
	//
	// Image magick command to create animated gif:
	//
	//     magick convert  -delay 6  -loop 0 -layers Optimize -compress LZW -strip -resize 240x180   frame*.png   Animation.gif
	//
	void SampleApp::ProcessCommandLine(const char* CmdLine)
	{
		const auto* pos = strchr(CmdLine, '-');
		while (pos != nullptr)
		{
			++pos;
			std::string Arg;
			if (!(Arg = GetArgument(pos, "mode")).empty())
			{
				if (StrCmpNoCase(Arg.c_str(), "D3D11", Arg.length()) == 0)
				{
				}
				else if (StrCmpNoCase(Arg.c_str(), "D3D12", Arg.length()) == 0)
				{
#if D3D12_SUPPORTED
					m_DeviceType = RENDER_DEVICE_TYPE_D3D12;
#else
                    LOG_ERROR_MESSAGE("Direct3D12 is not supported. Please select another device type");
#endif
				}
				else if (StrCmpNoCase(Arg.c_str(), "GL", Arg.length()) == 0)
				{
				}
				else if (StrCmpNoCase(Arg.c_str(), "GLES", Arg.length()) == 0)
				{
				}
				else if (StrCmpNoCase(Arg.c_str(), "VK", Arg.length()) == 0)
				{
				}
				else
				{
					LOG_ERROR_MESSAGE("Unknown device type: '", pos, "'. Only the following types are supported: D3D11, D3D12, GL, GLES, VK");
				}
			}
			else if (!(Arg = GetArgument(pos, "capture_path")).empty())
			{
				m_ScreenCaptureInfo.Directory = std::move(Arg);
				m_ScreenCaptureInfo.AllowCapture = true;
			}
			else if (!(Arg = GetArgument(pos, "capture_name")).empty())
			{
				m_ScreenCaptureInfo.FileName = std::move(Arg);
				m_ScreenCaptureInfo.AllowCapture = true;
			}
			else if (!(Arg = GetArgument(pos, "capture_fps")).empty())
			{
				m_ScreenCaptureInfo.CaptureFPS = atof(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "capture_frames")).empty())
			{
				m_ScreenCaptureInfo.FramesToCapture = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "capture_format")).empty())
			{
				if (StrCmpNoCase(Arg.c_str(), "jpeg", Arg.length()) == 0 || StrCmpNoCase(Arg.c_str(), "jpg", Arg.length()) == 0)
				{
					m_ScreenCaptureInfo.FileFormat = IMAGE_FILE_FORMAT_JPEG;
				}
				else if (StrCmpNoCase(Arg.c_str(), "png", Arg.length()) == 0)
				{
					m_ScreenCaptureInfo.FileFormat = IMAGE_FILE_FORMAT_PNG;
				}
				else
				{
					LOG_ERROR_MESSAGE("Unknown capture format. The following are allowed values: 'jpeg', 'jpg', 'png'");
				}
			}
			else if (!(Arg = GetArgument(pos, "capture_quality")).empty())
			{
				m_ScreenCaptureInfo.JpegQuality = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "capture_alpha")).empty())
			{
				m_ScreenCaptureInfo.KeepAlpha = (StrCmpNoCase(Arg.c_str(), "true", Arg.length()) == 0) || Arg == "1";
			}
			else if (!(Arg = GetArgument(pos, "width")).empty())
			{
				m_InitialWindowWidth = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "height")).empty())
			{
				m_InitialWindowHeight = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "validation")).empty())
			{
				m_ValidationLevel = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "adapter")).empty())
			{
				if (StrCmpNoCase(Arg.c_str(), "sw", Arg.length()) == 0)
				{
					m_AdapterType = ADAPTER_TYPE_SOFTWARE;
				}
				else
				{
					auto AdapterId = atoi(Arg.c_str());
					VERIFY_EXPR(AdapterId >= 0);
					m_AdapterId = static_cast<Uint32>(AdapterId >= 0 ? AdapterId : 0);
				}
			}
			else if (!(Arg = GetArgument(pos, "adapters_dialog")).empty())
			{
				m_bShowAdaptersDialog = (StrCmpNoCase(Arg.c_str(), "true", Arg.length()) == 0) || Arg == "1";
			}
			else if (!(Arg = GetArgument(pos, "show_ui")).empty())
			{
				m_bShowUI = (StrCmpNoCase(Arg.c_str(), "true", Arg.length()) == 0) || Arg == "1";
			}
			else if (!(Arg = GetArgument(pos, "golden_image_mode")).empty())
			{
			}
			else if (!(Arg = GetArgument(pos, "golden_image_tolerance")).empty())
			{
				m_GoldenImgPixelTolerance = atoi(Arg.c_str());
			}
			else if (!(Arg = GetArgument(pos, "vsync")).empty())
			{
				m_bVSync = (StrCmpNoCase(Arg.c_str(), "true", Arg.length()) == 0) || (StrCmpNoCase(Arg.c_str(), "on", Arg.length()) == 0) || Arg == "1";
			}
			else if (!(Arg = GetArgument(pos, "non_separable_progs")).empty())
			{
				m_bForceNonSeprblProgs = (StrCmpNoCase(Arg.c_str(), "true", Arg.length()) == 0) || (StrCmpNoCase(Arg.c_str(), "on", Arg.length()) == 0) || Arg == "1";
			}

			pos = strchr(pos, '-');
		}

		if (m_DeviceType == RENDER_DEVICE_TYPE_UNDEFINED)
		{
			SelectDeviceType();
			if (m_DeviceType == RENDER_DEVICE_TYPE_UNDEFINED)
			{
#if D3D12_SUPPORTED
				m_DeviceType = RENDER_DEVICE_TYPE_D3D12;
#elif VULKAN_SUPPORTED
                m_DeviceType = RENDER_DEVICE_TYPE_VULKAN;
#elif D3D11_SUPPORTED
                m_DeviceType = RENDER_DEVICE_TYPE_D3D11;
#elif GL_SUPPORTED || GLES_SUPPORTED
                m_DeviceType = RENDER_DEVICE_TYPE_GL;
#endif
			}
		}

		m_TheSample->ProcessCommandLine(CmdLine);
	}

	void SampleApp::WindowResize(int width, int height)
	{
		if (m_pSwapChain)
		{
			m_TheSample->PreWindowResize();
			m_pSwapChain->Resize(width, height);
			auto SCWidth = m_pSwapChain->GetDesc().Width;
			auto SCHeight = m_pSwapChain->GetDesc().Height;
			m_TheSample->WindowResize(SCWidth, SCHeight);
		}
	}

	void SampleApp::Update(double CurrTime, double ElapsedTime)
	{
		m_CurrentTime = CurrTime;

		if (m_pImGui)
		{
			const auto& SCDesc = m_pSwapChain->GetDesc();
			m_pImGui->NewFrame(SCDesc.Width, SCDesc.Height, SCDesc.PreTransform);
			if (m_bShowAdaptersDialog)
			{
				UpdateAdaptersDialog();
			}
		}
		if (m_pDevice)
		{
			m_TheSample->Update(CurrTime, ElapsedTime);
			m_TheSample->GetInputController().ClearState();
		}
	}

	void SampleApp::Render()
	{
		if (m_NumImmediateContexts == 0 || !m_pSwapChain)
			return;

		auto* pCtx = GetImmediateContext();
		auto* pRTV = m_pSwapChain->GetCurrentBackBufferRTV();
		auto* pDSV = m_pSwapChain->GetDepthBufferDSV();
		pCtx->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

		m_TheSample->Render();

		// Restore default render target in case the sample has changed it
		pCtx->SetRenderTargets(1, &pRTV, pDSV, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
		if (m_pImGui)
		{
			if (m_bShowUI)
			{
				// No need to call EndFrame as ImGui::Render calls it automatically
				m_pImGui->Render(pCtx);
			}
			else
			{
				m_pImGui->EndFrame();
			}
		}
	}

	void SampleApp::CompareGoldenImage(const std::string& FileName, ScreenCapture::CaptureInfo& Capture)
	{
		RefCntAutoPtr<Image> pGoldenImg;
		CreateImageFromFile(FileName.c_str(), &pGoldenImg, nullptr);
		if (!pGoldenImg)
		{
			LOG_ERROR_MESSAGE("Failed to load golden image from file ", FileName);
			m_ExitCode = -2;
			return;
		}

		const auto& TexDesc = Capture.pTexture->GetDesc();
		const auto& GoldenImgDesc = pGoldenImg->GetDesc();
		if (GoldenImgDesc.Width != TexDesc.Width)
		{
			LOG_ERROR_MESSAGE("Golden image width (", GoldenImgDesc.Width, ") does not match the captured image width (", TexDesc.Width, ")");
			m_ExitCode = -3;
			return;
		}
		if (GoldenImgDesc.Height != TexDesc.Height)
		{
			LOG_ERROR_MESSAGE("Golden image height (", GoldenImgDesc.Height, ") does not match the captured image height (", TexDesc.Height, ")");
			m_ExitCode = -4;
			return;
		}

		auto* const pCtx = GetImmediateContext();

		MappedTextureSubresource TexData;
		pCtx->MapTextureSubresource(Capture.pTexture, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, TexData);
		auto CapturedPixels = Image::ConvertImageData(TexDesc.Width, TexDesc.Height,
		                                              reinterpret_cast<const Uint8*>(TexData.pData), TexData.Stride,
		                                              TexDesc.Format, TEX_FORMAT_RGBA8_UNORM, false /*Keep alpha*/);
		pCtx->UnmapTextureSubresource(Capture.pTexture, 0, 0);

		auto* pGoldenImgPixels = reinterpret_cast<const Uint8*>(pGoldenImg->GetData()->GetDataPtr());

		m_ExitCode = 0;
		for (Uint32 row = 0; row < TexDesc.Height; ++row)
		{
			for (Uint32 col = 0; col < TexDesc.Width; ++col)
			{
				const auto* SrcPixel = &CapturedPixels[(col + row * TexDesc.Width) * 3];
				const auto* DstPixel = pGoldenImgPixels + row * GoldenImgDesc.RowStride + col * GoldenImgDesc.NumComponents;
				if (std::abs(int{SrcPixel[0]} - int{DstPixel[0]}) > m_GoldenImgPixelTolerance ||
					std::abs(int{SrcPixel[1]} - int{DstPixel[1]}) > m_GoldenImgPixelTolerance ||
					std::abs(int{SrcPixel[2]} - int{DstPixel[2]}) > m_GoldenImgPixelTolerance)
					++m_ExitCode;
			}
		}
	}

	void SampleApp::SaveScreenCapture(const std::string& FileName, ScreenCapture::CaptureInfo& Capture)
	{
		auto* const pCtx = GetImmediateContext();

		MappedTextureSubresource TexData;
		pCtx->MapTextureSubresource(Capture.pTexture, 0, 0, MAP_READ, MAP_FLAG_DO_NOT_WAIT, nullptr, TexData);
		const auto& TexDesc = Capture.pTexture->GetDesc();

		Image::EncodeInfo Info;
		Info.Width = TexDesc.Width;
		Info.Height = TexDesc.Height;
		Info.TexFormat = TexDesc.Format;
		Info.KeepAlpha = m_ScreenCaptureInfo.KeepAlpha;
		Info.pData = TexData.pData;
		Info.Stride = TexData.Stride;
		Info.FileFormat = m_ScreenCaptureInfo.FileFormat;
		Info.JpegQuality = m_ScreenCaptureInfo.JpegQuality;

		RefCntAutoPtr<IDataBlob> pEncodedImage;
		Image::Encode(Info, &pEncodedImage);
		pCtx->UnmapTextureSubresource(Capture.pTexture, 0, 0);

		FileWrapper pFile(FileName.c_str(), EFileAccessMode::Overwrite);
		if (pFile)
		{
			auto res = pFile->Write(pEncodedImage->GetDataPtr(), pEncodedImage->GetSize());
			if (!res)
			{
				LOG_ERROR_MESSAGE("Failed to write screen capture file '", FileName, "'.");
				m_ExitCode = -5;
			}
			pFile.Close();
		}
		else
		{
			LOG_ERROR_MESSAGE("Failed to create screen capture file '", FileName, "'. Verify that the directory exists and the app has sufficient rights to write to this directory.");
			m_ExitCode = -6;
		}
	}

	void SampleApp::Present()
	{
		if (!m_pSwapChain)
			return;

		auto* const pCtx = GetImmediateContext();

		if (m_pScreenCapture && m_ScreenCaptureInfo.FramesToCapture > 0)
		{
			if (m_CurrentTime - m_ScreenCaptureInfo.LastCaptureTime >= 1.0 / m_ScreenCaptureInfo.CaptureFPS)
			{
				pCtx->SetRenderTargets(0, nullptr, nullptr, RESOURCE_STATE_TRANSITION_MODE_NONE);
				m_pScreenCapture->Capture(m_pSwapChain, pCtx, m_ScreenCaptureInfo.CurrentFrame);

				m_ScreenCaptureInfo.LastCaptureTime = m_CurrentTime;

				--m_ScreenCaptureInfo.FramesToCapture;
				++m_ScreenCaptureInfo.CurrentFrame;
			}
		}

		m_pSwapChain->Present(m_bVSync ? 1 : 0);

		if (m_pScreenCapture)
		{
			while (auto Capture = m_pScreenCapture->GetCapture())
			{
				std::string FileName;
				{
					std::stringstream FileNameSS;
					if (!m_ScreenCaptureInfo.Directory.empty())
					{
						FileNameSS << m_ScreenCaptureInfo.Directory;
						if (m_ScreenCaptureInfo.Directory.back() != '/')
							FileNameSS << '/';
					}
					FileNameSS << m_ScreenCaptureInfo.FileName;

					FileNameSS << (m_ScreenCaptureInfo.FileFormat == IMAGE_FILE_FORMAT_JPEG ? ".jpg" : ".png");
					FileName = FileNameSS.str();
				}

				m_pScreenCapture->RecycleStagingTexture(std::move(Capture.pTexture));
			}
		}
	}
}
