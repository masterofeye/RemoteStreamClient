#pragma once

#include "IMP_Base.h"

namespace RW
{
	namespace IMP
	{
		namespace MERGE
		{
			typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
			{
			}tstMyInitialiseControlStruct;

			typedef struct stMyControlStruct : public CORE::tstControlStruct
			{
				std::vector<cInputBase*> *pvInput;
				cv::cuda::GpuMat *pOutput;
                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);

			}tstMyControlStruct;

			typedef struct stCropDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
			{
			}tstMyDeinitialiseControlStruct;

			class IMP_MergeFrames : public RW::CORE::AbstractModule
			{
				Q_OBJECT

			private:
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point					m_tStart;
#endif

				tenStatus ApplyMerge(cv::cuda::GpuMat *pgMat1, cv::cuda::GpuMat *pgMat2);

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
}