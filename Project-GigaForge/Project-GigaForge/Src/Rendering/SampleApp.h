#pragma once
#include "RenderingGlobals.h"
#include <memory>
#include <string>
#include <vector>
#include <DiligentCore/Platforms/Basic/interface/DebugUtilities.hpp>
#include "DeviceContext.h"

#include "DeviceContext.h"
#include "EngineFactory.h"
#include "DiligentTools/TextureLoader/interface/Image.h"
#include "RefCntAutoPtr.hpp"
#include "RenderDevice.h"
#include "SampleBase.h"
#include "ScreenCapture.hpp"
#include "SwapChain.h"

namespace Diligent
{

    class ImGuiImplDiligent;

    class SampleApp
    {
    public:
        SampleApp();
        ~SampleApp();
        virtual void        ProcessCommandLine(const char* CmdLine) final;
        virtual const char* GetAppTitle() const final { return m_AppTitle.c_str(); }
        virtual void        Update(double CurrTime, double ElapsedTime);
        virtual void        WindowResize(int width, int height);
        virtual void        Render();
        virtual void        Present();
        virtual void        SelectDeviceType() {};

        virtual void GetDesiredInitialWindowSize(int& width, int& height) final
        {
            width = m_InitialWindowWidth;
            height = m_InitialWindowHeight;
        }

     

        virtual int GetExitCode() const final
        {
            return m_ExitCode;
        }

        virtual bool IsReady() const final
        {
            return m_pDevice && m_pSwapChain && m_NumImmediateContexts > 0;
        }

        IDeviceContext* GetImmediateContext(size_t Ind = 0)
        {
            VERIFY_EXPR(Ind < m_NumImmediateContexts);
            return m_pDeviceContexts[Ind];
        }

    protected:
        void InitializeDiligentEngine(const NativeWindow* pWindow);
        void InitializeSample();
        void UpdateAdaptersDialog();

        virtual void SetFullscreenMode(const DisplayModeAttribs& DisplayMode)
        {
            m_bFullScreenMode = true;
            m_pSwapChain->SetFullscreenMode(DisplayMode);
        }
        virtual void SetWindowedMode()
        {
            m_bFullScreenMode = false;
            m_pSwapChain->SetWindowedMode();
        }

        void CompareGoldenImage(const std::string& FileName, ScreenCapture::CaptureInfo& Capture);
        void SaveScreenCapture(const std::string& FileName, ScreenCapture::CaptureInfo& Capture);

        RENDER_DEVICE_TYPE                         m_DeviceType = RENDER_DEVICE_TYPE_UNDEFINED;
        RefCntAutoPtr<IEngineFactory>              m_pEngineFactory;
        RefCntAutoPtr<IRenderDevice>               m_pDevice;
        std::vector<RefCntAutoPtr<IDeviceContext>> m_pDeviceContexts;
        Uint32                                     m_NumImmediateContexts = 0;
        RefCntAutoPtr<ISwapChain>                  m_pSwapChain;
        GraphicsAdapterInfo                        m_AdapterAttribs;
        std::vector<DisplayModeAttribs>            m_DisplayModes;

        std::unique_ptr<SampleBase> m_TheSample;

        int          m_InitialWindowWidth = 0;
        int          m_InitialWindowHeight = 0;
        int          m_ValidationLevel = -1;
        std::string  m_AppTitle;
        Uint32       m_AdapterId = 0;
        ADAPTER_TYPE m_AdapterType = ADAPTER_TYPE_UNKNOWN;
        std::string  m_AdapterDetailsString;
        int          m_SelectedDisplayMode = 0;
        bool         m_bVSync = false;
        bool         m_bFullScreenMode = false;
        bool         m_bShowAdaptersDialog = true;
        bool         m_bShowUI = true;
        bool         m_bForceNonSeprblProgs = false;
        double       m_CurrentTime = 0;
        Uint32       m_MaxFrameLatency = SwapChainDesc{}.BufferCount;

        // We will need this when we have to recreate the swap chain (on Android)
        SwapChainDesc m_SwapChainInitDesc;

        struct ScreenCaptureInfo
        {
            bool              AllowCapture = false;
            std::string       Directory;
            std::string       FileName = "frame";
            double            CaptureFPS = 30;
            double            LastCaptureTime = -1e+10;
            Uint32            FramesToCapture = 0;
            Uint32            CurrentFrame = 0;
            IMAGE_FILE_FORMAT FileFormat = IMAGE_FILE_FORMAT_PNG;
            int               JpegQuality = 95;
            bool              KeepAlpha = false;

        } m_ScreenCaptureInfo;
        std::unique_ptr<ScreenCapture> m_pScreenCapture;

        std::unique_ptr<ImGuiImplDiligent> m_pImGui;

        int             m_GoldenImgPixelTolerance = 0;
        int             m_ExitCode = 0;
    };

} // namespace Diligent
