
#include "VPL_FrameProcessor.hpp"
#include "VPL_Viewer.h"
#include "qbuffer.h"
#include "qboxlayout.h"
#include "qpushbutton.h"

#include <qvideowidget.h>
#include <qvideosurfaceformat.h>
#include "qobject.h"
#include "qslider.h"
#include "qlabel.h"
#include "qwidget.h"
#include "qstyle.h"

namespace RW
{
    namespace VPL
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



#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to Initialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
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
            QByteArray *pqbArray = new QByteArray();
            pqbArray->setRawData((char*)data->pstBitStream->pBuffer, data->pstBitStream->u32Size);
            QBuffer qBuffer(pqbArray);

            static RW::VPL::VPL_Viewer viewer;
            connect(this, SIGNAL(viewer.setVideoData(&qBuffer)), &viewer, SLOT(viewer.setVideoData(qBuffer)));

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to DoRender for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return enStatus;
        }

        tenStatus VPL_FrameProcessor::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenGraphic_Color");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to Deinitialise for nenPlayback_Simple module: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenSuccess;
        }

    }
}