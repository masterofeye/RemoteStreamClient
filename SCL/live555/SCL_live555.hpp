#pragma once

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <WinSock2.h>
#include <Ws2tcpip.h>

#include "GroupsockHelper.hh"
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

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
                VPL_Viewer *pViewer;
            }tstMyInitialiseControlStruct;

            typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
            {
                tstBitStream *pstBitStream;
                tstPayloadMsg stPayload;

                void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class SCL_live555 : public RW::CORE::AbstractModule
            {
                Q_OBJECT

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

