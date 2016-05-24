#pragma once

#include "AbstractModule.hpp"
#include <opencv2\videoio.hpp>

namespace RW
{
	namespace VG
	{
		typedef enum tenColorSpace
		{
			nenRGB,
			nenUnknown
		};

        typedef struct stVideoGrabberInitialiseControlStruct : public CORE::stInitialiseControlStruct
		{
			int nFrameWidth;
			int nFrameHeight;
			int nFPS;
			int nNumberOfFrames;
			tenColorSpace enColorSpace;
			std::string sFileName;
        }tstVideoGrabberInitialiseControlStruct;

        typedef struct stVideoGrabberControlStruct : public CORE::tstControlStruct
		{			
			void *pData;
			size_t nDataLength;
			int nCurrentFrameNumber;
			int nCurrentPositionMSec;
        }tstVideoGrabberControlStruct;

        typedef struct stVideoGrabberDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
        }tstVideoGrabberDeinitialiseControlStruct;

		class VideoGrabberSimu : public RW::CORE::AbstractModule
		{
			Q_OBJECT
		private:
			cv::VideoCapture m_videoCapture;
			int m_nFrameCounter;
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
