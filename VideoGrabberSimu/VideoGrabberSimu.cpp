#include "VideoGrabberSimu.hpp"

#include <opencv2\videoio.hpp>
#include "opencv2/opencv.hpp"
using namespace cv;

namespace RW
{
	namespace VG
	{
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
			m_Logger->debug("Initialise");
			if (pInitialiseControlStruct == NULL)
			{
				m_Logger->critical("VideoGrabberSimu::Initialise - pInitialiseControlStruct parameter is NULL");
				return tenStatus::nenError;
			}

			m_videoCapture.open(sFileName);
			if (m_videoCapture.isOpened())
			{
				m_Logger->info("The " + sFileName + " was opened succesfully");
				pInitialiseControlStruct->nFPS = m_videoCapture.get(CAP_PROP_FPS);
				pInitialiseControlStruct->nFrameHeight = m_videoCapture.get(CAP_PROP_FRAME_HEIGHT);
				pInitialiseControlStruct->nFrameWidth = m_videoCapture.get(CAP_PROP_FRAME_WIDTH);
				pInitialiseControlStruct->nNumberOfFrames = m_videoCapture.get(CAP_PROP_FRAME_COUNT);
				pInitialiseControlStruct->enColorSpace = CORE::nenRGB;
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
			if (pControlStruct == NULL)
			{
				m_Logger->critical("VideoGrabberSimu::DoRender - pControlStruct is NULL");
				return tenStatus::nenError;
			}

			if (pControlStruct->enCommand != CORE::tenGrabCommand)
			{
				m_Logger->error("VideoGrabberSimu::DoRender - wrong command");
				return tenStatus::nenError;
			}

			if (!m_videoCapture.isOpened())
			{
				m_Logger->error("VideoGrabberSimu::DoRender - video capture is not opened");
				return tenStatus::nenError;
			}
			
			Mat rawFrame;
			if (!m_videoCapture.read(rawFrame))
			{
				m_Logger->info("VideoGrabberSimu::DoRender - end of the file");
				return tenStatus::nenSuccess;
			}
			else if (pControlStruct->pData == NULL || pControlStruct->nDataLength == 0)
			{
				m_Logger->critical("VideoGrabberSimu::DoRender - pControlStruct->pData is NULL or pControlStruct->nDataLength is 0");
				return tenStatus::nenError;
			}
			else
			{
				Mat rgbFrame;
				cvtColor(rawFrame, rgbFrame, CV_BGR2RGB);
				size_t nFrameSize = rgbFrame.total() * rgbFrame.elemSize();
				size_t nActualDataLength = min(nFrameSize, pControlStruct->nDataLength);
				if (nActualDataLength < nFrameSize)
				{
					m_Logger->alert("VideoGrabberSimu::DoRender - requested data length is greater than the destination array size");
				}
				memcpy(pControlStruct->pData, rgbFrame.data, nActualDataLength);
				return tenStatus::nenSuccess;
			}
            m_Logger->debug("DoRender");
        }

        tenStatus VideoGrabberSimu::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            m_Logger->debug("Deinitialise");
			m_videoCapture.release();
            return tenStatus::nenSuccess;
        }
	} /*namespace VG*/
} /*namespace RW*/
