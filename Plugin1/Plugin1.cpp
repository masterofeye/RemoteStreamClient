#include "Plugin1.hpp"
#include <chrono>

namespace RW{
	namespace VG{
		Plugin1::Plugin1()
		{
			//_logger = spdlog::stdout_logger_mt("console");
		}


		Plugin1::~Plugin1()
		{
		}

		CORE::tstModuleVersion Plugin1::ModulVersion() {
			CORE::tstModuleVersion version = { 0, 1 };
			return version;
		}
        tenStatus Plugin1::Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) { 
            stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
            data->u8Test++;
            return tenStatus::nenError; }
        tenStatus Plugin1::DoRender(CORE::tstControlStruct * ControlStruct) { return tenStatus::nenError; }
        tenStatus Plugin1::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct){ return tenStatus::nenError; }

	}
}
