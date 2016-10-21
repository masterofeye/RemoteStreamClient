#include "ModuleFactory.hpp"
#include "Simu\VideoGrabberSimu.hpp"

namespace RW
{
	namespace VG
	{
		ModuleFactory::ModuleFactory() { }

		ModuleFactory::~ModuleFactory() { }	

        CORE::AbstractModule* ModuleFactory::Module(CORE::tenSubModule enModule)
		{
            CORE::AbstractModule* Module;
			tenStatus status = tenStatus::nenError;
			switch (enModule)
			{
			case CORE::tenSubModule::nenVideoGrabber_SIMU:
			    Module = new VGR::SIMU::VideoGrabberSimu(m_Logger);
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

        CORE::tenModule ModuleFactory::ModuleType()
        {
            return CORE::tenModule::enVideoGrabber;
        }
	} /*namespace VG*/
} /*namespace RW*/
