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

		void ModuleLoader::LoadPlugins(QList<AbstractModule *> *Pluginlist)
		{
            QDir pluginsDir = QDir(qApp->applicationDirPath());
			//pluginsDir.cd("Plugins");

            //Filter only the *.plu files
			QString qsSuffix = "*.plu";
            pluginsDir.setNameFilters(QStringList() << qsSuffix);

            //Any Files there?
            if (pluginsDir.entryList(QDir::Files).count() <= 0)
            {
                m_Logger->error("No plugins found. Path: ") << pluginsDir.absolutePath().toStdString();
            }
            //TODO Duplikate müssen gefiltert werden
			foreach(QString fileName, pluginsDir.entryList(QDir::Files))
            {
				QPluginLoader loader(pluginsDir.absoluteFilePath(fileName));
				if (loader.load())
                {
					QObject *plugin = loader.instance();
					if (plugin != NULL)
                    {
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
                                m_Logger->debug("nenVideoGrabber_SIMU loaded");
                                Pluginlist->append(module);
								break;
							case tenModule::enEncoder:
								module = moduleFactory->Module(tenSubModule::nenEncode_NVIDIA);
								if (module == nullptr)
									m_Logger->error("NVIDIA encoder module coudn't load correct.");
                                m_Logger->debug("nenEncode_NVIDIA loaded");
                                Pluginlist->append(module);
                                break; 
							case tenModule::enGraphic:
                                module = moduleFactory->Module(tenSubModule::nenGraphic_Color);
                                if (module == nullptr)
                                    m_Logger->error("nenGraphic_Color module coudn't load correct.");
                                else
                                {
                                    m_Logger->debug("nenGraphic_Color loaded");
                                    Pluginlist->append(module);
                                }

                                module = moduleFactory->Module(tenSubModule::nenGraphic_Crop);
                                if (module == nullptr)
                                    m_Logger->error("nenGraphic_Crop module coudn't load correct.");
                                else
                                {
                                    m_Logger->debug("nenGraphic_Crop loaded");
                                    Pluginlist->append(module);
                                }
                                module = moduleFactory->Module(tenSubModule::nenGraphic_Merge);
                                if (module == nullptr)
                                    m_Logger->error("nenGraphic_Merge module coudn't load correct.");
                                else
                                {
                                    m_Logger->debug("nenGraphic_Merge loaded");
                                    Pluginlist->append(module);
                                }
                                break;
							case tenModule::enDecoder:
								module = moduleFactory->Module(tenSubModule::nenDecoder_INTEL);
								if (module == nullptr)
									m_Logger->error("Intel Decode module coudn't load correct.");
                                m_Logger->debug("nenDecoder_INTEL loaded");
                                Pluginlist->append(module);
								break;
							case tenModule::enPlayback:
								module = moduleFactory->Module(tenSubModule::nenPlayback_Simple);
								if (module == nullptr)
									m_Logger->error("Qt Playback module coudn't load correct.");
                                m_Logger->debug("nenPlayback_Simple loaded");
                                Pluginlist->append(module);
								break;
							default:
								m_Logger->alert("This module wasn't found.");
								break;

							}
						}
					}
                }
			}
            m_Logger->debug("Amount of loaded modules: ") << Pluginlist->count();
		}
	}

}