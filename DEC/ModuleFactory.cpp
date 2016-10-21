#include "ModuleFactory.hpp"
#include "Intel\DEC_Intel.hpp"
#include "NVENC\DEC_NvDecodeD3D9.hpp"


namespace RW{
	namespace DEC{
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
            case CORE::tenSubModule::nenDecoder_INTEL:
                Module = new INTEL::DEC_Intel(m_Logger);
				if (Module != nullptr)
					status = tenStatus::nenSuccess;
				break;
            case CORE::tenSubModule::nenDecoder_NVIDIA:
                Module = new NVENC::CNvDecodeD3D9(m_Logger);
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
			return CORE::tenModule::enDecoder;
        }
	}
}

