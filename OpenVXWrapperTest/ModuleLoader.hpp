#pragma once
#include <QtCore>

namespace RW{
	namespace CORE{
		class AbstractModule;

		class ModuleLoader : public QObject
		{
			Q_OBJECT
		public:
			ModuleLoader();
			~ModuleLoader();

			void LoadPlugins(QList<AbstractModule const*> *Pluginlist);
		};
	}

}

