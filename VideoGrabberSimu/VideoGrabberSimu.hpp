#pragma once

#include "AbstractModule.hpp"
#include <opencv2/videoio.hpp>


namespace RW
{
	namespace VG
	{
        typedef struct stVideoGrabberInitialiseControlStruct : public CORE::tstInitialiseControlStruct
		{
			int nFrameWidth;
			int nFrameHeight;
			int nFPS;
			int nNumberOfFrames;
			std::string sFileName;
        }tstVideoGrabberInitialiseControlStruct;

        typedef struct  stVideoGrabberControlStruct : public CORE::tstControlStruct
		{
			cv::cuda::GpuMat *Output;
			int nCurrentFrameNumber;
			int nCurrentPositionMSec;
            REMOTE_API void UpdateData(CORE::tstControlStruct** Data, CORE::tenSubModule SubModuleType);
        }tstVideoGrabberControlStruct;

        typedef struct stVideoGrabberDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstVideoGrabberDeinitialiseControlStruct;

		class VideoGrabberSimu : public RW::CORE::AbstractModule
		{
			Q_OBJECT
		private:
			cv::VideoCapture m_videoCapture;

		public:
			explicit VideoGrabberSimu(std::shared_ptr<spdlog::logger> Logger);
			~VideoGrabberSimu();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;
		};
	} /*namespace VG*/
} /*namespace RW*/
