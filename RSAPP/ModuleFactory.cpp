#include "ModuleFactory.hpp"
#include "QT_simple\VPL_FrameProcessor.hpp"


namespace RW{
	namespace VPL{
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
			case CORE::tenSubModule::nenPlayback_Simple:
                Module = new VPL::QT_SIMPLE::VPL_FrameProcessor(m_Logger);
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
			return CORE::tenModule::enPlayback;
        }
	}
}

