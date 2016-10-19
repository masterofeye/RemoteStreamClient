#include "ModuleFactory.hpp"
#include "NVENC\ENC_CudaInterop.hpp"
#include "Intel\ENC_Intel.hpp"

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
				Module = new NVENC::ENC_CudaInterop(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
            case CORE::tenSubModule::nenEncode_INTEL:
                Module = new INTEL::ENC_Intel(m_Logger);
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

