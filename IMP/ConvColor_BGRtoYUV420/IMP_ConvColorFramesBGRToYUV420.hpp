#pragma once

#include "..\IMP_Base.h"

namespace RW
{
	namespace IMP
	{
        namespace COLOR_BGRTOYUV
		{
			typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
			{
			}tstMyInitialiseControlStruct;

			typedef struct stMyControlStruct : public CORE::tstControlStruct
			{
				cv::cuda::GpuMat *pData;
                RW::tstBitStream *pPayload;
                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);

			}tstMyControlStruct;

			typedef struct stCropDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
			{
			}tstMyDeinitialiseControlStruct;

			class IMP_ConvColorFramesBGRToYUV : public RW::CORE::AbstractModule
			{
				Q_OBJECT

			private:
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point					m_tStart;
#endif

			public:

				explicit IMP_ConvColorFramesBGRToYUV(std::shared_ptr<spdlog::logger> Logger);
				~IMP_ConvColorFramesBGRToYUV();
				virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
				virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
				virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
				virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
				virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

			};
		}
	}
}