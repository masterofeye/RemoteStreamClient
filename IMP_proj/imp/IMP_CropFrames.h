#pragma once

#include "IMP_Base.h"

namespace RW
{
	namespace IMP
	{
		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			bool bNeedConversion;
			cInputBase cInput;
		}tstMyInitialiseControlStruct;

		typedef struct stMyControlStruct : public CORE::tstControlStruct
		{
			stRectStruct stFrameRect;
		}tstMyControlStruct;

		typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
		{
			bool bNeedConversion;
			cOutputBase cOutput;
		}tstMyDeinitialiseControlStruct;

		class IMP_CropFrames : public RW::CORE::AbstractModule
		{
			Q_OBJECT


		private:
			cv::cuda::GpuMat m_cuMat;

		public:

			explicit IMP_CropFrames(std::shared_ptr<spdlog::logger> Logger);
			~IMP_CropFrames();
			virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
			virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

		};
	}
}