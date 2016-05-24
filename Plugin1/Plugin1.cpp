#include "Plugin1.hpp"
#ifdef TRACE_PERFORMANCE
#include "HighResolution\HighResClock.h"
#endif

namespace RW{
    namespace TEST{
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
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            stMyInitialiseControlStruct* data = static_cast<stMyInitialiseControlStruct*>(InitialiseControlStruct);
            data->u8Test++;
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Execution Time of Module Plugin1: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            m_Logger->debug("Initialise");
            return tenStatus::nenError; 
        }

        tenStatus Plugin1::DoRender(CORE::tstControlStruct * ControlStruct) 
        {
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock res;
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            stMyControlStruct* data = static_cast<stMyControlStruct*>(ControlStruct);
            data->u8Test++;

#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Execution Time of Module Plugin1: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            m_Logger->debug("DoRender");
            return tenStatus::nenError;
        }
        tenStatus Plugin1::Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct)
        {
#ifdef TRACE_PERFORMANCE

            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            stMyDeinitialiseControlStruct* data = static_cast<stMyDeinitialiseControlStruct*>(DeinitialiseControlStruct);
            data->u8Test++;
            m_Logger->debug("Deinitialise");
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            m_Logger->trace() << "Execution Time of Module Plugin1: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
            return tenStatus::nenError;
        }

	}
}
