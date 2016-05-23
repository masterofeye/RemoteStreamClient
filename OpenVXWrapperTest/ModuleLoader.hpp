#pragma once
#include <QtCore>
#include "spdlog\spdlog.h"
namespace RW{
	namespace CORE{
		class AbstractModule;

		class ModuleLoader : public QObject
		{
			Q_OBJECT
        private:
            std::shared_ptr<spdlog::logger> m_Logger;
		public:
            ModuleLoader(std::shared_ptr<spdlog::logger> Logger);
			~ModuleLoader();

			void LoadPlugins(QList<AbstractModule *> *Pluginlist);
		};
	}

}

