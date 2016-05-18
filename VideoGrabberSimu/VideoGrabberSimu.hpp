#pragma once

#include "AbstractModule.hpp"
#include <opencv2\videoio.hpp>

namespace RW
{
	namespace VG
	{
		class VideoGrabberSimu : public RW::CORE::AbstractModule
		{
			Q_OBJECT
		private:
			const std::string sFileName = "BR213_24bbp_10.avi";		//TODO: clarify how the input file name shall be provided
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
