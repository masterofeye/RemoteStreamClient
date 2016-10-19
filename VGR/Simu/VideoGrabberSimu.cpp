#include "VideoGrabberSimu.hpp"

#include <opencv2\videoio.hpp>
#include "opencv2\opencv.hpp"
#include "..\IMP\ConvColor_BGRtoNV12\IMP_ConvColorFramesBGRToNV12.hpp"
#include "..\IMP\Crop\IMP_CropFrames.hpp"
#include "..\IMP\Merge\IMP_MergeFrames.hpp"
#include "cuda_runtime_api.h"
#include "opencv2\cudev\common.hpp"
#include "HighResolution\HighResClock.h"
using namespace cv;

namespace RW
{
	namespace VGR
	{
        namespace SIMU
        {
            void stVideoGrabberControlStruct::UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType)
            {
                RW::tstPayloadMsg *pMsg = new RW::tstPayloadMsg;
                pMsg->u32FrameNbr = this->nCurrentFrameNumber;
                pMsg->u32Timestamp = this->nCurrentPositionMSec;
                RW::stBitStream *pPayload = new RW::tstBitStream;
                pPayload->u32Size = (uint32_t)sizeof(RW::stPayloadMsg);
                pPayload->pBuffer = (void*)pMsg;

                switch (SubModuleType)
                {
                case CORE::tenSubModule::nenGraphic_Crop:
                {
                    RW::IMP::CROP::tstMyControlStruct *data = static_cast<RW::IMP::CROP::tstMyControlStruct*>(*Data);
                    data->pInput = this->Output;
                    data->pvOutput = nullptr;

                    data->pPayload = pPayload;
                    break;
                }
                case CORE::tenSubModule::nenGraphic_Merge:
                {
                    RW::IMP::MERGE::tstMyControlStruct *data = static_cast<RW::IMP::MERGE::tstMyControlStruct*>(*Data);
                    if (!data->pvInput)
                        data->pvInput = new std::vector<cv::cuda::GpuMat*>();
                    data->pvInput->push_back(this->Output);
                    data->pOutput = nullptr;

                    data->pPayload = pPayload;
                    break;
                }
                case CORE::tenSubModule::nenGraphic_ColorBGRToNV12:
                {
                    auto* data = static_cast<RW::IMP::COLOR_BGRTONV12::tstMyControlStruct*>(*Data);
                    data->pData = Output;
                    data->pPayload = pPayload;
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
                //Initialize the Cuda Context
                cv::cuda::GpuMat test;
                test.create(1, 1, CV_8U);


                auto pControlStruct = (tstVideoGrabberInitialiseControlStruct*)pInitialiseControlStruct;
                m_pInit = pControlStruct;
                auto sFileName = pControlStruct->sFileName;
#if 1
				// File
                m_videoCapture.open(sFileName);
#else
				// Webcam
				m_videoCapture.open(0);
				sFileName = "Webcam";
#endif
				if (m_videoCapture.isOpened())
                {
                    m_Logger->info("The " + sFileName + " was opened succesfully");
                    pControlStruct->nFPS = m_videoCapture.get(CAP_PROP_FPS);
                    pControlStruct->nFrameHeight = m_videoCapture.get(CAP_PROP_FRAME_HEIGHT);
                    pControlStruct->nFrameWidth = m_videoCapture.get(CAP_PROP_FRAME_WIDTH);
                    pControlStruct->nNumberOfFrames = m_videoCapture.get(CAP_PROP_FRAME_COUNT);

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
#if 0
                    m_videoCapture.release();
                    if (Initialise(m_pInit) != tenStatus::nenSuccess){
                        m_Logger->error("VideoGrabberSimu::DoRender - re Initialise failed!");
                        return tenStatus::nenError;
                    }
                    if (DoRender(pControl) != tenStatus::nenSuccess){
                        m_Logger->error("VideoGrabberSimu::DoRender - re DoRender failed!");
                        return tenStatus::nenError;
                    }
#endif
//#ifdef TRACE_PERFORMANCE
//                    RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
//                    m_Logger->trace() << "DoRender time for module nenVideoGrabber_SIMU: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
//#endif
                    return tenStatus::nenError;
                }
                else
                {
                    cv::cuda::GpuMat *mat = new cv::cuda::GpuMat();
                    mat->upload(rawFrame);

					static int counter;
					WriteBufferToFile(rawFrame.ptr(), rawFrame.total() * rawFrame.elemSize(), "VGR_output", counter);

					pControl->Output = mat;

                    pControl->nCurrentFrameNumber = m_videoCapture.get(CAP_PROP_POS_FRAMES); // nFrameCounter++;
                    pControl->nCurrentPositionMSec = m_videoCapture.get(CAP_PROP_POS_MSEC);
#ifdef TRACE_PERFORMANCE
                    RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
                    m_Logger->trace() << "DoRender time for module nenVideoGrabber_SIMU: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                    m_Logger->info("Size of grabbed frame: ") << rawFrame.total() * rawFrame.channels() * rawFrame.elemSize1();
                    return tenStatus::nenSuccess;
                }


            }

            tenStatus VideoGrabberSimu::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
            {
                m_Logger->debug("Deinitialise nenVideoGrabber_SIMU");
                m_videoCapture.release();
                return tenStatus::nenSuccess;
            }
        }
	} /*namespace VGR*/
} /*namespace RW*/
