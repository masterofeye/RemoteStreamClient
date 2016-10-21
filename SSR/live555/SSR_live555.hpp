#pragma once

#ifndef SSR_LIVE555_HPP
#define SSR_LIVE555_HPP

#include "AbstractModule.hpp"

class RTPSink;
class MediaSource;
class RTCPInstance;
class RTSPServer;
class UsageEnvironment;
class Groupsock;
class FramedSource;
class ByteStreamMemoryBufferSource;
class ServerMediaSession;
struct in_addr;

namespace RW{
    namespace SSR{
		namespace LIVE555{

			typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
			{
				int iNumberOfSessions;
			}tstMyInitialiseControlStruct;

			typedef struct REMOTE_API stMyControlStruct : public CORE::tstControlStruct
			{
				tstBitStream *pstBitStream;

				void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
			}tstMyControlStruct;

			typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
			{
			}tstMyDeinitialiseControlStruct;

			//typedef struct trackState
			//{
			////	unsigned m_uTrackNumber;
			//	MediaSource* m_pSource;
			//	RTPSink* m_pSink;
			//	RTCPInstance* m_pRtcp;
			//	Groupsock *m_pRtpGroupsock;
			//	Groupsock *m_pRtcpGroupsock;
			//	ByteStreamMemoryBufferSource *m_pBufferSource;
			//};

			struct Session
			{
				unsigned m_uIndex;
				MediaSource* m_pSource;
				RTPSink* m_pSink;
				RTCPInstance* m_pRtcp;
				Groupsock *m_pRtpGroupsock;
				Groupsock *m_pRtcpGroupsock;
				ByteStreamMemoryBufferSource *m_pBufferSource;
				char m_cEventLoopBreaker;
			};

            class SSR_live555 : public RW::CORE::AbstractModule
            {
				Q_OBJECT

			private:
				UsageEnvironment* m_pEnv;
				char m_cEventLoopBreaker;
				static SSR_live555 *m_instance;
				//trackState* m_pTrackStates;
				std::vector<Session> m_vSessions;
				in_addr *m_pAvailableAddresses;
				int m_iNumberOfSessions;
				RTSPServer* m_pRTSPServer;
                RW::tstBitStream *m_pBitstream;

				static void afterPlaying(void* clientData);
				void InitialiseNewSession(ServerMediaSession *sms, int positionToInsert);
				void InitialiseGroupsocks(Session* session);
				void InitialiseSource(Session* session, stMyControlStruct* data);
				void InitialiseEnvironment();
				void ReportURL(ServerMediaSession* mediaSession);
				in_addr* GetAvailableAddresses();

            public:
                explicit SSR_live555(std::shared_ptr<spdlog::logger> Logger);
                ~SSR_live555();
                virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
                virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
                virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
                virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

            };

			
        }
    }
}

#endif //SSR_LIVE555_HPP