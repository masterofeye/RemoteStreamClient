#include "ModuleFactory.hpp"
#include "ENC_CudaInterop.hpp"


namespace RW{
	namespace ENC{
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
			case CORE::tenSubModule::nenEncode_NVIDIA:
				Module = new ENC::ENC_CudaInterop(m_Logger);
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
			return CORE::tenModule::enEncoder;
        }
	}
}

