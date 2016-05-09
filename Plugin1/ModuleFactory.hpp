#pragma once

#include "AbstractModuleFactory.hpp"

#include "Utils.h"

namespace RW
{
	namespace CORE
	{
		class AbstractModule;
	}

	namespace VG{

		class ModuleFactory : public CORE::AbstractModuleFactory
		{
			Q_OBJECT
				Q_PLUGIN_METADATA(IID "AbstractModuleFactory" FILE "Plugin1.json")
                Q_INTERFACES(RW::CORE::AbstractModuleFactory)

		public:
			ModuleFactory();
			~ModuleFactory();
            CORE::AbstractModule* Module(CORE::tenSubModule enModule);
		};
	}
}



