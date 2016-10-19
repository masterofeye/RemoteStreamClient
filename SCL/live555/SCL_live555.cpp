
#include "SCL_live555.hpp"
#include "..\..\DEC\INTEL\DEC_Intel.hpp"
#include "..\..\DEC\NVENC\DEC_NvDecodeD3D9.hpp"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <WinSock2.h>
#include <Ws2tcpip.h>

#include "GroupsockHelper.hh"
#include <liveMedia.hh>
#include <BasicUsageEnvironment.hh>

#include "C:\Projekte\RemoteStreamClient\SCL\DummySink.h"

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

//#ifdef TRACE_PERFORMANCE
//#include "HighResolution\HighResClock.h"
//#endif

namespace RW
{
    namespace SCL
    {
        namespace LIVE555
        {
			RW::CORE::HighResClock::time_point t1;

			SCL_live555* SCL_live555::m_instance;

            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                switch (SubModuleType)
                {
                case CORE::tenSubModule::nenDecoder_INTEL:
                {
                    RW::DEC::INTEL::tstMyControlStruct *data = static_cast<RW::DEC::INTEL::tstMyControlStruct*>(*Data);
                    data->pstEncodedStream = (this->pstBitStream);
                    break;
                }
                case CORE::tenSubModule::nenDecoder_NVIDIA:
                {
                    RW::DEC::NVENC::tstMyControlStruct *data = static_cast<RW::DEC::NVENC::tstMyControlStruct*>(*Data);
                    data->pstEncodedStream = (this->pstBitStream);
                    break;
                }
                default:
                    break;
                }
            }

            SCL_live555::SCL_live555(std::shared_ptr<spdlog::logger> Logger)
                : RW::CORE::AbstractModule(Logger)
            {
				SCL_live555::m_instance = this;
            }

            SCL_live555::~SCL_live555()
            {
            }

            CORE::tstModuleVersion SCL_live555::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule SCL_live555::SubModulType()
            {
                return CORE::tenSubModule::nenReceive_Simple;
            }

            tenStatus SCL_live555::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("Initialise nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
                stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("Initialise: Data of stMyInitialiseControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
				if (!m_bIsInitialised)
					InitialiseSession();

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

			void SCL_live555::InitialiseSession()
			{
				InitialiseGroupsocks();
				sessionState.source = H264VideoRTPSource::createNew(*m_pEnv, sessionState.m_pRtpGroupsock, 96);
				const unsigned estimatedSessionBandwidth = 500;
				const unsigned maxCNAMElen = 100;
				unsigned char CNAME[maxCNAMElen + 1];
				gethostname((char*)CNAME, maxCNAMElen);
				CNAME[maxCNAMElen] = '\0';
				sessionState.rtcpInstance
					= RTCPInstance::createNew(*m_pEnv, sessionState.m_pRtcpGroupsock,
					estimatedSessionBandwidth, CNAME,
					NULL /* we're a client */, sessionState.source);
				sessionState.sink = DummySink::createNew(*m_pEnv, &SCL_live555::GetDataFromSink, this);
				m_bIsInitialised = true;
			}

			void SCL_live555::InitialiseGroupsocks()
			{
				TaskScheduler* scheduler = BasicTaskScheduler::createNew();
				m_pEnv = BasicUsageEnvironment::createNew(*scheduler);
				char const* sessionAddressStr = "232.255.42.42";
				const unsigned short rtpPortNum = 8888;
				const unsigned short rtcpPortNum = rtpPortNum + 1;
				struct in_addr sessionAddress;
				sessionAddress.s_addr = our_inet_addr(sessionAddressStr);
				const Port rtpPort(rtpPortNum);
				const Port rtcpPort(rtcpPortNum);
				char* sourceAddressStr = "aaa.bbb.ccc.ddd";
				struct in_addr sourceFilterAddress;
				sourceFilterAddress.s_addr = our_inet_addr(sourceAddressStr);
				sessionState.m_pRtpGroupsock = new Groupsock(*m_pEnv, sessionAddress, sourceFilterAddress, rtpPort); //todo: fix memory leaks
				sessionState.m_pRtcpGroupsock = new  Groupsock(*m_pEnv, sessionAddress, sourceFilterAddress, rtcpPort);
				sessionState.m_pRtcpGroupsock->changeDestinationParameters(sourceFilterAddress, 0, ~0);
			}

            tenStatus SCL_live555::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("DoRender nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                t1 = RW::CORE::HighResClock::now();
#endif

				dataControlStruct = static_cast<stMyControlStruct*>(ControlStruct);
				if (dataControlStruct == nullptr)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
				if (!dataControlStruct->pstBitStream)
					dataControlStruct->pstBitStream = new RW::tstBitStream;  //todo: fix a memory leak
				
				bool res = StartReceiving();
				if (res == false)
				{
					m_Logger->error("DoRender: startPlaying failed!");
					return tenStatus::nenError;
				}

				m_cEventLoopBreaker = 0;
				m_pEnv->taskScheduler().doEventLoop(&m_cEventLoopBreaker);

				if (dataControlStruct->pstBitStream->pBuffer == NULL)
				{
					m_Logger->error("DoRender: buffer is empty!");
					return tenStatus::nenError;
				}

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif

                return enStatus;
            }

			bool SCL_live555::StartReceiving()
			{
				unsigned *size = &(dataControlStruct->pstBitStream->u32Size);
				//sessionState.sink = DummySink::createNew(*m_pEnv, &SCL_live555::GetDataFromSink, this);
				bool res = sessionState.sink->startPlaying(*sessionState.source, afterPlaying, dataControlStruct->pstBitStream->pBuffer);
				return res;
			}

			void SCL_live555::GetDataFromSink(void* clientData, u_int8_t* buffer, unsigned size)
			{
				((SCL_live555*)clientData)->GetDataFromSink(buffer, size);
			}

			void SCL_live555::GetDataFromSink(u_int8_t* buffer, unsigned size)
			{
				dataControlStruct->pstBitStream->pBuffer = (uint8_t*)buffer;
				dataControlStruct->pstBitStream->u32Size = size;
				m_cEventLoopBreaker = ~0;
				sessionState.sink->stopPlaying();
			}

			void SCL_live555::afterPlaying(void *clientData)
			{
				
				int clientIndex = (int)clientData;
				auto instance = m_instance;
				Medium::close(instance->sessionState.sink);
				instance->m_cEventLoopBreaker = ~0;
				
			}

            tenStatus SCL_live555::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif


				Medium::close(sessionState.rtcpInstance);
				//Medium::close(sessionState.source);


				sessionState.m_pRtpGroupsock->removeAllDestinations();
				delete sessionState.m_pRtpGroupsock;
				delete sessionState.m_pRtcpGroupsock;
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Deinitialise for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}