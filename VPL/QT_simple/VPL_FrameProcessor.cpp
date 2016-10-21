
#include "VPL_FrameProcessor.hpp"
#include "VPL_Viewer.hpp"

namespace RW
{
    namespace VPL
    {
        namespace QT_SIMPLE
        {
            void stMyControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
            }

            VPL_FrameProcessor::VPL_FrameProcessor(std::shared_ptr<spdlog::logger> Logger)
                : RW::CORE::AbstractModule(Logger)
            {

            }

            VPL_FrameProcessor::~VPL_FrameProcessor()
            {
            }

            CORE::tstModuleVersion VPL_FrameProcessor::ModulVersion() {
                CORE::tstModuleVersion version = { 0, 1 };
                return version;
            }

            CORE::tenSubModule VPL_FrameProcessor::SubModulType()
            {
                return CORE::tenSubModule::nenPlayback_Simple;
            }

            tenStatus VPL_FrameProcessor::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;

                m_Logger->debug("Initialise nenPlayback_Simple");
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
                if (data->pViewer == nullptr)
                {
                    m_Logger->error("Initialise: data->pVPL_Viewer of stMyInitialiseControlStruct is empty!");
                    enStatus = tenStatus::nenError;
                    return enStatus;
                }

                data->pViewer->connectToViewer(this);

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Initialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus VPL_FrameProcessor::DoRender(CORE::tstControlStruct * ControlStruct)
            {
                tenStatus enStatus = tenStatus::nenSuccess;
                m_Logger->debug("DoRender nenPlayback_Simple");
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

                emit FrameBufferChanged(data->pstBitStream);

                if (data->pPayload)
                {
                    if (data->pPayload->pBuffer){
                        RW::stPayloadMsg *pMsg = (RW::stPayloadMsg *)data->pPayload->pBuffer;
                        uint32_t u32Timestamp = pMsg->u32Timestamp;
                        uint32_t u32FrameNbr = pMsg->u32FrameNbr;

                        SAFE_DELETE_ARRAY(data->pPayload->pBuffer);
                    }
                    SAFE_DELETE(data->pPayload);
                }

                static int count;
                WriteBufferToFile(data->pstBitStream->pBuffer, data->pstBitStream->u32Size, "Client_VPL", count);

                //// only used for Qt::QueuedConnection since this is enableing assynchronous threads using copies of the parameters
                //SAFE_DELETE_ARRAY(data->pstBitStream->pBuffer);
                //SAFE_DELETE(data->pstBitStream);

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to DoRender for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return enStatus;
            }

            tenStatus VPL_FrameProcessor::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenPlayback_Simple");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                m_Logger->trace() << "Time to Deinitialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                return tenStatus::nenSuccess;
            }
        }
    }
}