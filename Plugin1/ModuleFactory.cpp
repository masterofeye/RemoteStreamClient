#include "ModuleFactory.hpp"
#include "Plugin1.hpp"


namespace RW{
	namespace VG{
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
			case CORE::tenSubModule::nenVideoGrabber_SIMU:
			    Module = new VG::Plugin1();
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
			case CORE::tenSubModule::nenVideoGrabber_FG_USB:
				break;
			default:
				//TODO Status can't find module
				status = tenStatus::nenError;
				break;
			}
            return Module;

		}
	}
}

