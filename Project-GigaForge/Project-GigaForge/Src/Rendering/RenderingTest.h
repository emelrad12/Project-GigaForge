#pragma once
#include "RenderingGlobals.h"
#include "SampleBase.h"
void RenderingTest();
namespace Diligent
{
	class Tutorial01_HelloTriangle final : public SampleBase
	{
	public:
		virtual void Initialize(const SampleInitInfo& InitInfo) override final;

		virtual void Render() override final;
		virtual void Update(double CurrTime, double ElapsedTime) override final;

		virtual const Char* GetSampleName() const override final { return "Tutorial01: Hello Triangle"; }

	private:
		RefCntAutoPtr<IPipelineState> m_pPSO;
	};
}
