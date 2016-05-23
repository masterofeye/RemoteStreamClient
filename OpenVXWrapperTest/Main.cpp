#include <iostream>
#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"
#include "Plugin1.hpp"
#include "GraphBuilder.h"
#include "ENC_CudaInterop.hpp"
#include "HighResolution\HighResClock.h"
#include "spdlog\spdlog.h"

#include <QApplication>

#define DEBUG 1
#define TRACE_PERFORMANCE

void logtest()
{
    auto logger = spdlog::get("file_logger");
    logger->info("Log from application");
}



int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

#ifdef DEBUG
        spdlog::set_level(spdlog::level::debug);
#elif TRACE
        spdlog::set_level(spdlog::level::trace);
#else
        spdlog::set_level(spdlog::level::info);
#endif


        auto file_logger = spdlog::rotating_logger_mt("file_logger", (qApp->applicationDirPath() + "/logfile.log").toStdString(), 1048576 * 5, 3);
        file_logger->debug("******************");
        file_logger->debug("*Applicationstart*");
        file_logger->debug("******************");
        try
        {
            RW::tenStatus status = RW::tenStatus::nenError;
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
            RW::CORE::ModuleLoader ml(file_logger);

            /*Load Plugins*/
            QList<RW::CORE::AbstractModule *> list;
            ml.LoadPlugins(&list);
#ifdef TRACE_PERFORMANCE
            RW::CORE::HighResClock::time_point t2 = RW::CORE::HighResClock::now();
            file_logger->trace() << "Time to load Plugins: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
            t1 = RW::CORE::HighResClock::now();
#endif
            RW::CORE::Context context(file_logger);
            if (context.IsInitialized())
            {
                RW::CORE::Graph graph(&context, file_logger);
                RW::CORE::KernelManager kernelManager(&context, file_logger);

                RW::CORE::GraphBuilder builder(&list, file_logger, &graph, &context);

                RW::VG::tstMyInitialiseControlStruct initControl;
                RW::VG::tstMyControlStruct control;
                RW::VG::tstMyDeinitialiseControlStruct deinitControl;

                builder.BuildNode(&kernelManager, &initControl, &control, &deinitControl, RW::CORE::tenSubModule::nenVideoGrabber_SIMU);
                builder.BuildNode(&kernelManager, &initControl, &control, &deinitControl, RW::CORE::tenSubModule::nenGraphic_Color);
                builder.BuildNode(&kernelManager, &initControl, &control, &deinitControl, RW::CORE::tenSubModule::nenEncode_NVIDIA);

                if (graph.VerifyGraph() == RW::tenStatus::nenSuccess)
                {
                    if (graph.ScheduleGraph() == RW::tenStatus::nenSuccess)
                    {

                        file_logger->debug("******************");
                        file_logger->debug("*Graph excecution*");
                        file_logger->debug("******************");
#ifdef TRACE_PERFORMANCE
                        t2 = RW::CORE::HighResClock::now();
                        file_logger->trace() << "Prepare Graph: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
                        t1 = RW::CORE::HighResClock::now();
#endif
                        graph.WaitGraph();
#ifdef TRACE_PERFORMANCE
                        t2 = RW::CORE::HighResClock::now();
                        file_logger->trace() << "Graph execution: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
#endif
                    }
                }
            }
        }
        catch (...)
        {
            file_logger->flush();
        }
    return 0;
}