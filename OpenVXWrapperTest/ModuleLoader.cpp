#include "ModuleLoader.hpp"
#include "AbstractModuleFactory.hpp"
#include "AbstractModule.hpp"

#include <QList>
#include <QDebug>

namespace RW{
	namespace CORE{
        ModuleLoader::ModuleLoader(std::shared_ptr<spdlog::logger> Logger)
            : m_Logger(Logger)
		{
		}


		ModuleLoader::~ModuleLoader()
		{
		}

		void ModuleLoader::LoadPlugins(QList<AbstractModule const*> *Pluginlist)
		{
            //TODO Dynamic Link
            QDir pluginsDir = QDir("C:\\Projekte\\RemoteRepros\\RemoteStreamClient\\build\\x64\\Debug");

			pluginsDir.cd("Plugins");
			foreach(QString fileName, pluginsDir.entryList(QDir::Files)) {
                qDebug() << fileName;
				QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
                if (loader.load())
                {
                    QObject *plugin = loader.instance();
                    if (plugin != NULL) {
                        AbstractModuleFactory *moduleFactory = NULL;
                        moduleFactory = qobject_cast <AbstractModuleFactory*>(plugin);
                        if (moduleFactory != NULL)
                        {
                            moduleFactory->SetLogger(m_Logger);
                            AbstractModule *module = nullptr;
                            switch (moduleFactory->ModuleType())
                            {

                            case tenModule::enVideoGrabber:
                                module = moduleFactory->Module(tenSubModule::nenVideoGrabber_SIMU);
                                if (module == nullptr)
                                    m_Logger->error("Virtual video source module coudn't load correct.");
                                break;
                            case tenModule::enEncoder:
                                module = moduleFactory->Module(tenSubModule::nenEncode_NVIDIA);
                                if (module == nullptr)
                                    m_Logger->error("NVIDIA encoder module coudn't load correct.");
                                break;
                            case tenModule::enGraphic:
                                module = moduleFactory->Module(tenSubModule::nenGraphic_Color);
                                if (module == nullptr)
                                    m_Logger->error("Colorspace conversion module coudn't load correct.");
                                break;
                            case tenModule::enDecoder:
                                module = moduleFactory->Module(tenSubModule::nenDecoder_INTEL);
                                if (module == nullptr)
                                    m_Logger->error("Intel Decode module coudn't load correct.");
                                break;
                            case tenModule::enPlayback:
                                module = moduleFactory->Module(tenSubModule::nenPlayback_Simple);
                                if (module == nullptr)
                                    m_Logger->error("Qt Playback module coudn't load correct.");
                                break;
                            default:
                                m_Logger->alert("No module found.");
                                break;

                            }

                            Pluginlist->append(module);
                        }
                    }
                }
			}
		}
	}

}