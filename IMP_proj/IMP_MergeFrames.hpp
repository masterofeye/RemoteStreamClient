#pragma once

#include "IMP_Base.h"

namespace RW
{
	namespace IMP
	{

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