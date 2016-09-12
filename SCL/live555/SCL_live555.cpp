
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

#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW
{
    namespace SCL
    {
        namespace LIVE555
        {
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


                m_iCount = 0;




#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus SCL_live555::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("DoRender nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

                stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
                if (data == nullptr)
                {
                    m_Logger->error("DoRender: Data of stMyControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }

                char* filename = new char[64];
                sprintf(filename, "c:\\dummy\\Server_NVENC\\SSR_output_%04d.264", m_iCount);
                FILE *pFile;
                fopen_s(&pFile, filename, "rb");
                if (!pFile)
                {
                    m_Logger->error("SCL_live555::DoRender: File did not load!");
                    return tenStatus::nenError;
                }
                // obtain file size:
                fseek(pFile, 0, SEEK_END);
                // Size of one frame
                long lSize = ftell(pFile);
                rewind(pFile);

                m_Logger->info("SCL_live555::DoRender: File size to read in: ") << lSize;

                // allocate memory to contain the whole file:
                uint8_t *buffer = (uint8_t*)malloc(sizeof(uint8_t)*lSize);
                // copy the file into the buffer:
                size_t result = fread(buffer, sizeof(uint8_t), lSize, pFile);
                if (!buffer)
                {
                    m_Logger->error("SCL_live555::DoRender: Empty buffer!");
                    return tenStatus::nenError;
                }
                fclose(pFile);

                data->pstBitStream = new RW::tstBitStream();
                data->pstBitStream->pBuffer = buffer;
                data->pstBitStream->u32Size = lSize;

                m_iCount++;

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus SCL_live555::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenReceive_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif







#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Deinitialise for nenReceive_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}