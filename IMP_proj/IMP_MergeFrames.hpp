#pragma once

#include "IMP_Base.h"

namespace RW
{
	namespace IMP
	{
		typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			bool bNeedConversion;
			cInputBase cInput1;
			cInputBase cInput2;
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

		class IMP_MergeFrames : public RW::CORE::AbstractModule
		{
			Q_OBJECT


		private:
			cv::cuda::GpuMat m_cuMat1;
			cv::cuda::GpuMat m_cuMat2;

			tenStatus ApplyMerge(cv::cuda::GpuMat gMat1, cv::cuda::GpuMat gMat2, cv::cuda::GpuMat *pgMat);

		public:

			explicit IMP_MergeFrames(std::shared_ptr<spdlog::logger> Logger);
			~IMP_MergeFrames();
			virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
			virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

		};
	}
}