#include "ModuleFactory.hpp"
#include "Crop\IMP_CropFrames.hpp"
#include "ConvColor_BGRtoYUV420\IMP_ConvColorFramesBGRToYUV420.hpp"
#include "ConvColor_NV12toRGB\IMP_ConvColorFramesNV12ToRGB.hpp"
#include "Merge\IMP_MergeFrames.hpp"


namespace RW{
	namespace IMP{
		ModuleFactory::ModuleFactory()
		{
		}


		ModuleFactory::~ModuleFactory()
		{
		}

        CORE::AbstractModule* ModuleFactory::Module(CORE::tenSubModule enModule)
		{
            CORE::AbstractModule* Module;
			tenStatus status = tenStatus::nenError;
			switch (enModule)
			{
			case CORE::tenSubModule::nenGraphic_Crop:
				Module = new IMP::CROP::IMP_CropFrames(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
			case CORE::tenSubModule::nenGraphic_Merge:
				Module = new IMP::MERGE::IMP_MergeFrames(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
			case CORE::tenSubModule::nenGraphic_ColorBGRToYUV:
				Module = new IMP::COLOR_BGRTOYUV::IMP_ConvColorFramesBGRToYUV(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
            case CORE::tenSubModule::nenGraphic_ColorNV12ToRGB:
                Module = new IMP::COLOR_NV12TORGB::IMP_ConvColorFramesNV12ToRGB(m_Logger);
                if (Module != nullptr)
                    status = tenStatus::nenSuccess;
                break;
            default:
				//TODO Status can't find module
				status = tenStatus::nenError;
				break;
			}
            return Module;

		}

        CORE::tenModule ModuleFactory::ModuleType()
        {
			return CORE::tenModule::enGraphic;
        }
	}
}

