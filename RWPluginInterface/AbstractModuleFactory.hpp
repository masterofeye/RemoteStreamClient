#pragma once
#include "Utils.h"
#include <QtPlugin>
#include "spdlog\spdlog.h"

namespace RW
{
	namespace CORE
	{
		class AbstractModule;

		class AbstractModuleFactory : public QObject
		{

			Q_OBJECT
        protected:
            std::shared_ptr<spdlog::logger> m_Logger;
		public:
			virtual ~AbstractModuleFactory(){}
            virtual CORE::AbstractModule* Module(CORE::tenSubModule enModule) = 0;
            inline void SetLogger(std::shared_ptr<spdlog::logger> Logger){ m_Logger = Logger;}


		};


	}

}

QT_BEGIN_NAMESPACE
#define FilterInterface_iid "AbstractModuleFactory"
Q_DECLARE_INTERFACE(RW::CORE::AbstractModuleFactory, FilterInterface_iid)
QT_END_NAMESPACE
