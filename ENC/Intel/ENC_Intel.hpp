#pragma once

#include "AbstractModule.hpp"

namespace RW{
    namespace ENC{
        namespace INTEL{

            struct sInputParams;
            class CEncodingPipeline;

            typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
            {
                sInputParams *pParams;  
            }tstMyInitialiseControlStruct;

            typedef struct stMyControlStruct : public CORE::tstControlStruct
            {
                tstBitStream *pInput;
                tstBitStream *pstBitStream;
                tstBitStream *pPayload;

                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class ENC_Intel : public RW::CORE::AbstractModule
            {
                Q_OBJECT

            private:
                sInputParams *m_pParams;
                std::auto_ptr<CEncodingPipeline>  m_pPipeline;


            public:
                explicit ENC_Intel(std::shared_ptr<spdlog::logger> Logger);
                ~ENC_Intel();
                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

            };
        }
    }
}

