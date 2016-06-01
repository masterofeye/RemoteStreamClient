#include "ModuleFactory.hpp"
#include "Plugin1.hpp"


namespace RW{
    namespace TEST{
		ModuleFactory::ModuleFactory()
		{
		}


		ModuleFactory::~ModuleFactory()
		{
		}

        CORE::AbstractModule* ModuleFactory::Module(CORE::tenSubModule enModule)
		{
            CORE::AbstractModule* Module = nullptr;
			tenStatus status = tenStatus::nenError;
			switch (enModule)
			{
            case CORE::tenSubModule::nenTest_Test:
                Module = new TEST::Plugin1(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
			//case CORE::tenSubModule::nenVideoGrabber_FG_USB:
			//	break;
			default:
				//TODO Status can't find module
				status = tenStatus::nenError;
				break;
			}
            return Module;

		}

        CORE::tenModule ModuleFactory::ModuleType()
        {
            return CORE::tenModule::enTest;
        }
	}
}

