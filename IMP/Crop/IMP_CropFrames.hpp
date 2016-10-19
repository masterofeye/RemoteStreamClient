#pragma once

#include "..\IMP_Base.h"
#include "..\..\CSR\Config.h"

namespace RW
{
	namespace IMP
	{
		namespace CROP
		{
			typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
			{
				std::vector<cv::Rect> vFrameRect;
			}tstMyInitialiseControlStruct;

			typedef struct stMyControlStruct : public CORE::tstControlStruct
			{
				cv::cuda::GpuMat *pInput;
                RW::tstBitStream *pPayload;
				std::vector<cv::cuda::GpuMat*> *pvOutput;
                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);

			}tstMyControlStruct;

			typedef struct stCropDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
			{
			}tstMyDeinitialiseControlStruct;

			class IMP_CropFrames : public RW::CORE::AbstractModule
			{
				Q_OBJECT

			private:
				std::vector<cv::Rect> m_vRect;
#ifdef TRACE_PERFORMANCE
				RW::CORE::HighResClock::time_point					m_tStart;
#endif

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
}