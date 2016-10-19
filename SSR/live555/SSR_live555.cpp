
#include "SSR_live555.hpp"

//#define WIN32_LEAN_AND_MEAN

#include "GroupsockHelper.hh"
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>

#include "AvailableAddressesProvider.h"
#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <Windows.h>

//#include "UsageEnvironment.hh"

//#pragma comment (lib, "Ws2_32.lib")
//#pragma comment (lib, "Mswsock.lib")
//#pragma comment (lib, "AdvApi32.lib")

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

#define INDICATOR 4D0E75A3-C76A-44B4-BDE2-CA8EC4C6F00C
#define MAX_SIZE 100000
#define PAYLOAD_FORMAT 96

namespace RW
{
    namespace SSR
    {
        namespace LIVE555
        {
            SSR_live555* SSR_live555::m_instance;


            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                //SAFE_DELETE(this->pstBitStream);
            }

            SSR_live555::SSR_live555(std::shared_ptr<spdlog::logger> Logger)
                : RW::CORE::AbstractModule(Logger)
            {
                SSR_live555::m_instance = this;
                m_pBitstream = nullptr;
                m_pEnv = nullptr;
                m_cEventLoopBreaker = 0;
                m_pAvailableAddresses = nullptr;
                m_iNumberOfSessions = 0;
                m_pRTSPServer = nullptr;
            }

            SSR_live555::~SSR_live555()
            {
            }

            CORE::tstModuleVersion SSR_live555::ModulVersion()
            {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule SSR_live555::SubModulType()
            {
                return CORE::tenSubModule::nenStream_Simple;
            }

            tenStatus SSR_live555::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("Initialise nenPlayback_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point timeBefore = RW::CORE::HighResClock::now();
#endif
                stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("Initialise: Data of stMyInitialiseControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }

                InitialiseEnvironment();

                m_iNumberOfSessions = 1;

                m_pAvailableAddresses = GetAvailableAddresses();	//todo: can be a local variable

                //m_pTrackStates = new trackState[m_iNumberOfSessions];


                if (m_pRTSPServer == NULL)
                {
                    m_Logger->error("Failed to create RTSP server:  %s \n", m_pEnv->getResultMsg());
                    exit(1);
                }
                ServerMediaSession* mediaSession
                    = ServerMediaSession::createNew(*m_pEnv, "testStream", "",
                    "Session streamed by \"testH265VideoStreamer\"",
                    True /*SSM*/);

                for (int i = 0; i < m_iNumberOfSessions; i++)
                {
                    InitialiseNewSession(mediaSession, i);
                }

                m_pRTSPServer->addServerMediaSession(mediaSession);
                ReportURL(mediaSession);

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point timeAfter = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(timeBefore, timeAfter).count() << "ms.";
#endif
                return enStatus;
            }

            void SSR_live555::ReportURL(ServerMediaSession* mediaSession)
            {
                char* url = m_pRTSPServer->rtspURL(mediaSession);
                //*m_pEnv << "Play this stream using the URL \"" << url << "\"\n";
                delete[] url;
            }

            void SSR_live555::InitialiseEnvironment()
            {
                const Port port = 8554;
                TaskScheduler* scheduler = BasicTaskScheduler::createNew();
                m_pEnv = BasicUsageEnvironment::createNew(*scheduler);
                m_pRTSPServer = RTSPServer::createNew(*m_pEnv, port);
            }

            in_addr* SSR_live555::GetAvailableAddresses() //todo: fix everything
            {
                AvailableAddressesProvider provider = AvailableAddressesProvider(m_iNumberOfSessions);
                return provider.GetAllAvailableAddresses(m_pEnv);
            }

            void SSR_live555::InitialiseNewSession(ServerMediaSession *sms, int positionToInsert)
            {

                Session session = Session();
                session.m_uIndex = positionToInsert;

                InitialiseGroupsocks(&session);
                OutPacketBuffer::maxSize = MAX_SIZE;	//todo: _probably_ error handling is needed

                session.m_pSink = H264VideoRTPSink::createNew(*m_pEnv, session.m_pRtpGroupsock, PAYLOAD_FORMAT);
                const unsigned estimatedSessionBandwidth = 500;
                const unsigned maxCNAMElen = 100;
                unsigned char CNAME[maxCNAMElen + 1];
                gethostname((char*)CNAME, maxCNAMElen);
                CNAME[maxCNAMElen] = '\0';

                session.m_pRtcp = RTCPInstance::createNew(*m_pEnv, session.m_pRtcpGroupsock,
                    estimatedSessionBandwidth, CNAME,
                    session.m_pSink, NULL /* we're a server */, True);
                sms->addSubsession(PassiveServerMediaSubsession::createNew(*(session.m_pSink), session.m_pRtcp));
                m_vSessions.push_back(session);
            }

            void SSR_live555::InitialiseGroupsocks(Session* session)
            {
                struct in_addr destinationAddress;
                destinationAddress.s_addr = our_inet_addr("232.255.42.42");//= m_pAvailableAddresses[positionToInsert];
                const unsigned short rtpPortNum = 8888 + (session->m_uIndex * 2);
                const unsigned short rtcpPortNum = rtpPortNum + 1;
                const unsigned char ttl = 255;
                const Port rtpPort(rtpPortNum);
                const Port rtcpPort(rtcpPortNum);
                session->m_pRtpGroupsock = new Groupsock(*m_pEnv, destinationAddress, rtpPort, ttl);
                session->m_pRtcpGroupsock = new Groupsock(*m_pEnv, destinationAddress, rtcpPort, ttl);
                session->m_pRtpGroupsock->multicastSendOnly();
                session->m_pRtcpGroupsock->multicastSendOnly();
            }

            tenStatus SSR_live555::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("DoRender nenStream_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                if (m_pBitstream){
                    if (m_pBitstream->pBuffer){
                        delete[] m_pBitstream->pBuffer;
                        m_pBitstream->pBuffer = nullptr;
                    }
                    SAFE_DELETE(m_pBitstream);
                }

                stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
                if (data->pstBitStream == nullptr)
                {
                    m_Logger->error("DoRender: data->pstBitStream is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
                if (data->pstBitStream->pBuffer == nullptr)
                {
                    m_Logger->error("DoRender: data->pstBitStream->pBuffer is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }

                for (Session& session : m_vSessions)
                {
                    InitialiseSource(&session, data);

                    //*m_pEnv << "Beginning to read from file...\n";
                    bool retval = session.m_pSink->startPlaying(*(session.m_pSource), SSR_live555::afterPlaying, (void*)&session);

                    if (!retval)
                    {
                        m_Logger->error("DoRender startPlaying failed");
                        return tenStatus::nenError;
                    }

                    session.m_cEventLoopBreaker = 0;

                    m_pEnv->taskScheduler().doEventLoop(&(session.m_cEventLoopBreaker));
                }

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
                auto t3 = std::chrono::high_resolution_clock::now();
                m_Logger->trace() << "Time of sending BYE: " << std::chrono::high_resolution_clock::to_time_t(t3);
#endif
                return enStatus;
            }

            void SSR_live555::InitialiseSource(Session* session, stMyControlStruct* data)
            {
                session->m_pBufferSource =
                    ByteStreamMemoryBufferSource::createNew(*m_pEnv, (uint8_t*)data->pstBitStream->pBuffer, data->pstBitStream->u32Size, false);
                session->m_pSource = H264VideoStreamDiscreteFramer::createNew(*m_pEnv, session->m_pBufferSource);
            }

            void SSR_live555::afterPlaying(void *clientData)
            {
                Session* session = (Session*)clientData;
                //*(instance->m_pEnv) << "...done reading from file\n";
                session->m_pSink->stopPlaying();
                Medium::close(session->m_pSource);
                session->m_cEventLoopBreaker = ~0;
            }

            tenStatus SSR_live555::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                for (Session& session : m_vSessions)
                {
                    //Medium::close(m_pTrackStates[i].m_pSink);
                    Medium::close(session.m_pRtcp);
                    session.m_pRtpGroupsock->removeAllDestinations();
                    delete session.m_pRtpGroupsock;
                    delete session.m_pRtcpGroupsock;
                }

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Deinitialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}