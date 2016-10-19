#pragma once

#include "AbstractModule.hpp"
#include "qbuffer.h"

namespace RW{
    namespace VPL{
        namespace QT_SIMPLE{
            class VPL_Viewer;

            typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
            {
                VPL_Viewer *pViewer;
            }tstMyInitialiseControlStruct;

            typedef struct stMyControlStruct : public CORE::tstControlStruct
            {
                tstBitStream *pstBitStream;
                tstBitStream *pPayload;
                void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class VPL_FrameProcessor : public RW::CORE::AbstractModule
            {
                Q_OBJECT

            public:
                explicit VPL_FrameProcessor(std::shared_ptr<spdlog::logger> Logger);
                ~VPL_FrameProcessor();
                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

            signals:
                void FrameBufferChanged(void *pBuffer);

            };
        }
    }
}

