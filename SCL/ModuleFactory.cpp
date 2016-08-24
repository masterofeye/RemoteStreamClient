#include "ModuleFactory.hpp"
#include "live555\SCL_live555.hpp"


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
            case CORE::tenSubModule::nenReceive_Simple:
                Module = new SCL::LIVE555::SCL_live555(m_Logger);
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
			return CORE::tenModule::enReceive;
        }
	}
}

