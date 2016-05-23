#pragma once

#include "AbstractModuleFactory.hpp"

#include "Utils.h"

namespace RW
{
    namespace CORE
	{
		class AbstractModule;
	}

	namespace VPL{

		class ModuleFactory : public CORE::AbstractModuleFactory
		{
			Q_OBJECT
				Q_PLUGIN_METADATA(IID "AbstractModuleFactory" FILE "VideoPlayer.json")
                Q_INTERFACES(RW::CORE::AbstractModuleFactory)

		public:
			ModuleFactory();
			~ModuleFactory();
            CORE::AbstractModule* Module(CORE::tenSubModule enModule);
            CORE::tenModule ModuleType();
		};
	}
}



