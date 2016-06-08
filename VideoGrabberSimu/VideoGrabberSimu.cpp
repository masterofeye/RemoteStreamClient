#include "VideoGrabberSimu.hpp"

#include <opencv2\videoio.hpp>
#include "opencv2/opencv.hpp"
#include "..\IMP_proj\IMP.h"
#include "HighResolution\HighResClock.h"
using namespace cv;

namespace RW
{
	namespace VG
	{
  


        void stVideoGrabberControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
        {
            switch (SubModuleType)
            {
            case CORE::tenSubModule::nenGraphic_Color:
            case CORE::tenSubModule::nenGraphic_Crop:
            {
                RW::IMP::tstMyControlStruct *data = static_cast<RW::IMP::tstMyControlStruct*>(*Data);
                data->pcInput->_pvImg = pOutputData->pBuffer;
                break;
            }
			case CORE::tenSubModule::nenGraphic_Merge:
			{
				RW::IMP::tstMyControlStruct *data = static_cast<RW::IMP::tstMyControlStruct*>(*Data);
				if (data->pcInput->_bSetImg2)
				{
					data->pcInput->_pInput2->_pvImg = pOutputData->pBuffer;
				}
				else 
				{
					data->pcInput->_pInput1->_pvImg = pOutputData->pBuffer;
				}
				break;
			}
			default:
                break;
            }
        }

        VideoGrabberSimu::VideoGrabberSimu(std::shared_ptr<spdlog::logger> Logger) : RW::CORE::AbstractModule(Logger) { }

		VideoGrabberSimu::~VideoGrabberSimu() { }

		CORE::tstModuleVersion VideoGrabberSimu::ModulVersion() 
		{
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

        CORE::tenSubModule VideoGrabberSimu::SubModulType()
        {
            return CORE::tenSubModule::nenVideoGrabber_SIMU;
        }

        tenStatus VideoGrabberSimu::Initialise(CORE::tstInitialiseControlStruct * pInitialiseControlStruct)
		{

			m_Logger->debug("Initialise nenVideoGrabber_SIMU");
			if (pInitialiseControlStruct == NULL)
			{
				m_Logger->critical("VideoGrabberSimu::Initialise - pInitialiseControlStruct parameter is NULL");
				return tenStatus::nenError;
			}
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif

			auto pControlStruct = (tstVideoGrabberInitialiseControlStruct*)pInitialiseControlStruct;
			auto sFileName = pControlStruct->sFileName;
			m_videoCapture.open(sFileName);
			if (m_videoCapture.isOpened())
			{
				m_Logger->info("The " + sFileName + " was opened succesfully");
				pControlStruct->nFPS = m_videoCapture.get(CAP_PROP_FPS);
				pControlStruct->nFrameHeight = m_videoCapture.get(CAP_PROP_FRAME_HEIGHT);
                pControlStruct->nFrameWidth = m_videoCapture.get(CAP_PROP_FRAME_WIDTH);
				pControlStruct->nNumberOfFrames = m_videoCapture.get(CAP_PROP_FRAME_COUNT);
				//m_nFrameCounter = 0;
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "Time to load Plugins: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return tenStatus::nenSuccess;
			}
			else
			{
				m_Logger->critical("Cannot open " + sFileName);
				return tenStatus::nenError;
			}

        }

        tenStatus VideoGrabberSimu::DoRender(CORE::tstControlStruct * pControlStruct) 
        {
            m_Logger->debug("DoRender: nenVideoGrabber_SIMU");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
			if (pControlStruct == NULL)
			{
				m_Logger->critical("VideoGrabberSimu::DoRender - pControlStruct is NULL");
				return tenStatus::nenError;
			}

			auto pControl = (stVideoGrabberControlStruct*)pControlStruct;

			if (!m_videoCapture.isOpened())
			{
				m_Logger->error("VideoGrabberSimu::DoRender - video capture is not opened");
				return tenStatus::nenError;
			}
			
			//static int nFrameCounter = 0;

			Mat rawFrame;
			if (!m_videoCapture.read(rawFrame))
			{
				m_Logger->info("VideoGrabberSimu::DoRender - end of the file");
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "DoRender time for module nenVideoGrabber_SIMU: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return tenStatus::nenSuccess;
			}
			else if (pControl->pOutputData->pBuffer == NULL)
			{
				m_Logger->critical("VideoGrabberSimu::DoRender - pControlStruct->pData is NULL");
				return tenStatus::nenError;
			}
			else
			{
				size_t nFrameSize = rawFrame.total() * rawFrame.elemSize();
                pControl->pOutputData->pBuffer = (void*)rawFrame.data;
				pControl->pOutputData->u32Size = nFrameSize;
				pControl->nCurrentFrameNumber = m_videoCapture.get(CAP_PROP_POS_FRAMES); // nFrameCounter++;
                pControl->nCurrentPositionMSec = m_videoCapture.get(CAP_PROP_POS_MSEC);
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
				m_Logger->trace() << "DoRender time for module nenVideoGrabber_SIMU: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
				return tenStatus::nenSuccess;
			}


        }

        tenStatus VideoGrabberSimu::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise nenVideoGrabber_SIMU");
			m_videoCapture.release();
            return tenStatus::nenSuccess;
        }
	} /*namespace VG*/
} /*namespace RW*/
