#pragma once

#include "AbstractModule.hpp"

class UsageEnvironment;
class RTPSource;
class MediaSink;
class RTCPInstance;
class Groupsock;
class RTSPClient;

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

            typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
            {
                tstBitStream *pstBitStream;

                void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
            }tstMyControlStruct;

            typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
            {
            }tstMyDeinitialiseControlStruct;

            class SCL_live555 : public RW::CORE::AbstractModule
            {
                Q_OBJECT

			private:
				UsageEnvironment* m_pEnv;
				static SCL_live555 *m_instance;
				bool m_bIsInitialised = false;
				char m_cEventLoopBreaker;
				RTSPClient* ourRTSPClient;
				stMyControlStruct* dataControlStruct;
				void InitialiseSession();
				void InitialiseGroupsocks();
				bool StartReceiving();

				static void afterPlaying(void* clientData);

            public:
				struct sessionState_t {
					RTPSource* source;
					MediaSink* sink;
					RTCPInstance* rtcpInstance;
					Groupsock *m_pRtpGroupsock;
					Groupsock *m_pRtcpGroupsock;
				} sessionState;
				static void GetDataFromSink(void* clientData, uint8_t* buffer, unsigned size);
				void GetDataFromSink(uint8_t* buffer, unsigned size);
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

