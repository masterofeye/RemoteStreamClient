#pragma once
#include "Utils.h"
#include <QtPlugin>
namespace RW
{
	namespace CORE
	{
		class AbstractModule;

		class AbstractModuleFactory : public QObject
		{
			Q_OBJECT
		public:
			virtual ~AbstractModuleFactory(){}
            virtual CORE::AbstractModule* Module(CORE::tenSubModule enModule) = 0;


		};


	}

}

QT_BEGIN_NAMESPACE
#define FilterInterface_iid "AbstractModuleFactory"
Q_DECLARE_INTERFACE(RW::CORE::AbstractModuleFactory, FilterInterface_iid)
QT_END_NAMESPACE
