#pragma once

#include "AbstractModule.hpp"

namespace RW
{
    namespace SCL
    {
        namespace LIVE555
        {

            class VPL_Viewer;

            typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
            {
            }tstMyInitialiseControlStruct;

            typedef struct stMyControlStruct : public CORE::tstControlStruct
            {
                tstBitStream *pstBitStream;

                REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class SCL_live555 : public RW::CORE::AbstractModule
            {
                Q_OBJECT

            private:
                int m_iCount;

            public:
                explicit SCL_live555(std::shared_ptr<spdlog::logger> Logger);
                ~SCL_live555();
                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;
            };
        }
    }
}

