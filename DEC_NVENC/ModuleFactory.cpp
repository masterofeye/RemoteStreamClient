#include "ModuleFactory.hpp"


namespace RW
{
	namespace DEC
	{
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
			case CORE::tenSubModule::nenDecoder_NVIDIA:
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

