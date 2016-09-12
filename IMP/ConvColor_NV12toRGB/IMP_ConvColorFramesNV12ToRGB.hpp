#pragma once

#include "..\IMP.h"

namespace RW{
    namespace IMP{
        namespace COLOR_NV12TORGB{

            typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
            {
                uint32_t nWidth;
                uint32_t nHeight;
            }tstMyInitialiseControlStruct;

            typedef struct stMyControlStruct : public CORE::tstControlStruct
            {
                RW::IMP::tstInputOutput *pInput;
                tstBitStream *pPayload;
                tstBitStream *pOutput;
                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);

            }tstMyControlStruct;

            typedef struct stCropDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class IMP_ConvColorFramesNV12ToRGB : public RW::CORE::AbstractModule
            {
                Q_OBJECT

            public:
                explicit IMP_ConvColorFramesNV12ToRGB(std::shared_ptr<spdlog::logger> Logger);
                ~IMP_ConvColorFramesNV12ToRGB();
                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

            private:
                uint32_t m_u32Width;
                uint32_t m_u32Height;

            };
        }
    }
}

