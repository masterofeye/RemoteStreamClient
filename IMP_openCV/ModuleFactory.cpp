#include "ModuleFactory.hpp"
#include "IMP_CropFrames.hpp"
#include "IMP_ConvColorFrames.hpp"
#include "IMP_MergeFrames.hpp"
#include "IMP_ConvColorFramesYUV420ToRGB.hpp"


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
			case CORE::tenSubModule::nenGraphic_Color:
				Module = new IMP::COLOR::IMP_ConvColorFrames(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
            case CORE::tenSubModule::nenGraphic_ColorYUV420ToRGB:
                Module = new IMP::COLOR_YUV420TORGB::IMP_ConvColorFramesYUV420ToRGB(m_Logger);
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

