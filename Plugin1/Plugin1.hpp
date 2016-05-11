#pragma once

#include <QtCore>
#include <QtPlugin>
#include "AbstractModule.hpp"
#include "Utils.h"

namespace RW
{
	namespace VG{

        typedef struct stMyInitialiseControlStruct : public CORE::tstInitialiseControlStruct
        {
            uint u8Test;
        }tstMyInitialiseControlStruct;

        typedef struct stMyControlStruct : public CORE::tstControlStruct
        {
            uint u8Test;
        }tstMyControlStruct;

        typedef struct stMyDeinitialiseControlStruct : public CORE::tstDeinitialiseControlStruct
        {
            uint u8Test;
        }tstMyDeinitialiseControlStruct;

		class Plugin1 : public RW::CORE::AbstractModule
		{
			Q_OBJECT
		private:

		public:

            explicit Plugin1(std::shared_ptr<spdlog::logger> Logger);
			~Plugin1();
            virtual CORE::tenSubModule SubModulType() Q_DECL_OVERRIDE;
			virtual CORE::tstModuleVersion ModulVersion() Q_DECL_OVERRIDE;
            virtual tenStatus Initialise(CORE::tstInitialiseControlStruct * InitialiseControlStruct) Q_DECL_OVERRIDE;
			virtual tenStatus DoRender(CORE::tstControlStruct * ControlStruct) Q_DECL_OVERRIDE;
            virtual tenStatus Deinitialise(CORE::tstDeinitialiseControlStruct *DeinitialiseControlStruct) Q_DECL_OVERRIDE;

		};
	}
}
