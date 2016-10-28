
#include "SCL_live555.hpp"
#include "..\..\DEC\INTEL\DEC_Intel.hpp"
#include "..\..\DEC\NVENC\DEC_NvDecodeD3D9.hpp"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <WinSock2.h>
#include <Ws2tcpip.h>

#include "..\DummySink.h"

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

namespace RW
{
    namespace SCL
    {
        namespace LIVE555
        {
			RW::CORE::HighResClock::time_point t1;

			SCL_live555* SCL_live555::m_pInstance;

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
				SCL_live555::m_pInstance = this;
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
					vInitialiseSession();

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

			void SCL_live555::vInitialiseSession()
			{
				vInitialiseGroupsocks();
				sessionState.m_pSource = H264VideoRTPSource::createNew(*m_pEnv, sessionState.m_pRtpGroupsock, 96); //todo: check for memory leaks
				const unsigned estimatedSessionBandwidth = 500;
				const unsigned maxCNAMElen = 100;
				unsigned char CNAME[maxCNAMElen + 1];  //todo: fix the name
				gethostname((char*)CNAME, maxCNAMElen);
				CNAME[maxCNAMElen] = '\0';  //todo: fix forming of the zero-terminated string
				sessionState.m_pRtcpInstance
					= RTCPInstance::createNew(*m_pEnv, sessionState.m_pRtcpGroupsock,
					estimatedSessionBandwidth, CNAME,
					NULL /* we're a client */, sessionState.m_pSource);
				//sessionState.sink = DummySink::createNew(*m_pEnv, &SCL_live555::GetDataFromSink, this);
				m_bIsInitialised = true;
			}

			void SCL_live555::vInitialiseGroupsocks()
			{
				TaskScheduler* scheduler = BasicTaskScheduler::createNew();
				m_pEnv = BasicUsageEnvironment::createNew(*scheduler);  //todo: check it, memory leaks are possible
				char const* sessionAddressStr = "232.255.42.42";
				const unsigned short rtpPortNum = 8888;
				const unsigned short rtcpPortNum = rtpPortNum + 1;
				struct in_addr sessionAddress;
				sessionAddress.s_addr = inet_addr(sessionAddressStr);
				const Port rtpPort(rtpPortNum);
				const Port rtcpPort(rtcpPortNum);
				char* sourceAddressStr = "aaa.bbb.ccc.ddd";
				struct in_addr sourceFilterAddress;
				sourceFilterAddress.s_addr = inet_addr(sourceAddressStr);
				sessionState.m_pRtpGroupsock = new Groupsock(*m_pEnv, sessionAddress, sourceFilterAddress, rtpPort);
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

				m_pDataControlStruct = static_cast<stMyControlStruct*>(ControlStruct);
				if (m_pDataControlStruct == nullptr)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }
				//if (!m_pDataControlStruct->pstBitStream) //may cause troubles
					m_pDataControlStruct->pstBitStream = new RW::tstBitStream;  //todo: fix a memory leak
				
				bool res = boStartReceiving();
				if (res == false)
				{
					m_Logger->error("DoRender: startPlaying failed!");
					return tenStatus::nenError;
				}

				m_cEventLoopBreaker = 0;
				m_pEnv->taskScheduler().doEventLoop(&m_cEventLoopBreaker);

				if (m_pDataControlStruct->pstBitStream->pBuffer == NULL)
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

			bool SCL_live555::boStartReceiving()
			{
				unsigned *size = &(m_pDataControlStruct->pstBitStream->u32Size);
				sessionState.m_pSink = DummySink::createNew(*m_pEnv, &SCL_live555::vGetDataFromSink, this);
				bool res = sessionState.m_pSink->startPlaying(*sessionState.m_pSource, afterPlaying, m_pDataControlStruct->pstBitStream->pBuffer);
				return res;
			}

			void SCL_live555::vGetDataFromSink(void* clientData, u_int8_t* buffer, unsigned size)
			{
				((SCL_live555*)clientData)->vGetDataFromSink(buffer, size);
			}

			void SCL_live555::vGetDataFromSink(u_int8_t* buffer, unsigned size)
			{
				m_pDataControlStruct->pstBitStream->pBuffer = (uint8_t*)buffer;
				m_pDataControlStruct->pstBitStream->u32Size = size;
				m_cEventLoopBreaker = ~0;
				sessionState.m_pSink->stopPlaying();
			}

			void SCL_live555::afterPlaying(void *clientData)
			{
				
				int clientIndex = (int)clientData;
				auto instance = m_pInstance;
				Medium::close(instance->sessionState.m_pSink);
				instance->m_cEventLoopBreaker = ~0;
				
			}

            tenStatus SCL_live555::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif


				Medium::close(sessionState.m_pRtcpInstance);
				Medium::close(sessionState.m_pSink);
				//Medium::close(sessionState.m_pSource);

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