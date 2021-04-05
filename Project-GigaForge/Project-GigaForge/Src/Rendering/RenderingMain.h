#pragma once
#include <d3d12.h>
#include <string>
#include <windows.h>
#include <wrl/client.h>


#include "GameTimer.h"

namespace Rendering
{
	class Renderer
	{
	public:
		bool Init();
		int Run();
		void Update(const GameTimer& gt);
		void OnResize();
		LRESULT MsgProc(HWND hwnd, UINT msg, WPARAM uint, LPARAM long_);
		HINSTANCE hinstance;
		static Renderer* mApp;
		static Renderer& GetApp();
		int mClientWidth = 800;
		int mClientHeight = 600;
		HWND      mhMainWnd;
		std::wstring mMainWndCaption = L"d3d App";
		Microsoft::WRL::ComPtr<ID3D12Device> md3dDevice;

		bool      mAppPaused = false;  // is the application paused?
		bool      mMinimized = false;  // is the application minimized?
		bool      mMaximized = false;  // is the application maximized?
		bool      mResizing = false;   // are the resize bars being dragged?
		bool      mFullscreenState = false;// fullscreen enabled
		GameTimer mTimer;
	};
}
