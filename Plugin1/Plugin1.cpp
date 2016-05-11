#include "Plugin1.hpp"
#include <chrono>

namespace RW{
	namespace VG{
        Plugin1::Plugin1(std::shared_ptr<spdlog::logger> Logger) :
            RW::CORE::AbstractModule(Logger)
		{
		}


		Plugin1::~Plugin1()
		{
		}

		CORE::tstModuleVersion Plugin1::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}

        CORE::tenSubModule Plugin1::SubModulType()
        {
            return CORE::tenSubModule::nenVideoGrabber_SIMU;
        }
        tenStatus Plugin1::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct)
        { 
            stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
            data->u8Test++;
            m_Logger->debug("Initialise");
            return tenStatus::nenError; 
        }

        tenStatus Plugin1::DoRender(CORE::tstControlStruct * ControlStruct) 
        {
            stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
            data->u8Test++;
            m_Logger->debug("DoRender");
            return tenStatus::nenError;
        }
        tenStatus Plugin1::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
            stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);
            data->u8Test++;
            m_Logger->debug("Deinitialise");
            return tenStatus::nenError;
        }

	}
}
