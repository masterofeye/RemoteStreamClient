#include "ModuleLoader.hpp"
#include "AbstractModuleFactory.hpp"
#include "Interface.hpp"

#include <QList>
#include <QDebug>

namespace RW{
	namespace CORE{
		ModuleLoader::ModuleLoader()
		{
		}


		ModuleLoader::~ModuleLoader()
		{
		}

		void ModuleLoader::LoadPlugins(QList<AbstractModule const*> *Pluginlist)
		{

            QDir pluginsDir = QDir("C:\\Projekte\\RemoteRepros\\RemoteStreamClient\\build\\x64");

            qDebug() << pluginsDir.dirName();
#if defined(Q_OS_WIN)
			if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
				pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
			if (pluginsDir.dirName() == "MacOS") {
				pluginsDir.cdUp();
				pluginsDir.cdUp();
				pluginsDir.cdUp();
			}
#endif
			pluginsDir.cd("plugins");

			foreach(QString fileName, pluginsDir.entryList(QDir::Files)) {
                qDebug() << fileName;
				QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
                if (loader.load())
                {
                    //auto t1 = HighResClock::now();
                    QObject *plugin = loader.instance();
                    //auto t2 = HighResClock::now();
                    //auto t3 = HighResClock::diff_milli(t1, t2);
                    //qDebug() << t3.count();
                    if (plugin != NULL) {
                        AbstractModuleFactory *plugin2 = NULL;
                        plugin2 = qobject_cast <AbstractModuleFactory*>(plugin);
                        if (plugin2 != NULL)
                        {

                            AbstractModule * module2 = plugin2->Module(tenSubModule::nenVideoGrabber_SIMU);
                            Pluginlist->append(module2);
                        }
                    }
                }
			}
		}
	}

}