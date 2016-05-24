#pragma once

#include "AbstractModuleFactory.hpp"

namespace RW
{
	namespace VG{

		class ModuleFactory : public CORE::AbstractModuleFactory
		{
			Q_OBJECT
				Q_PLUGIN_METADATA(IID "AbstractModuleFactory" FILE "VideoGrabberSimu.json")
                Q_INTERFACES(RW::CORE::AbstractModuleFactory)

		public:
			ModuleFactory();
			~ModuleFactory();
            CORE::AbstractModule* Module(CORE::tenSubModule enModule);
            CORE::tenModule ModuleType();
		};
	}
}
