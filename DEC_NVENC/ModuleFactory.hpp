#pragma once

#include "AbstractModuleFactory.hpp"
#include "Utils.h"


namespace RW
{
	namespace CORE
	{
		class AbstractModule;
	}

	namespace DEC
	{
		class ModuleFactory : public CORE::AbstractModuleFactory
		{
			Q_OBJECT
				Q_PLUGIN_METADATA(IID "AbstractModuleFactory" FILE "DEC.json")
				Q_INTERFACES(RW::CORE::AbstractModuleFactory)

		public:
			ModuleFactory();
			~ModuleFactory();
			CORE::AbstractModule* Module(CORE::tenSubModule enModule);
			CORE::tenModule ModuleType();
		};
	}
}