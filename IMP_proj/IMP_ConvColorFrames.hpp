#pragma once

#include "IMP_Base.h"

namespace RW
{
	namespace IMP
	{
		class IMP_ConvColorFrames : public RW::CORE::AbstractModule
		{
			Q_OBJECT

		private:
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point					m_tStart;
#endif

		public:

			explicit IMP_ConvColorFrames(std::shared_ptr<spdlog::logger> Logger);
			~IMP_ConvColorFrames();
			virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
			virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

		};

	}
}