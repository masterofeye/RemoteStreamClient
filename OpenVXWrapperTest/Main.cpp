#include <iostream>
#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"
#include "Plugin1.hpp"
#include "HighResolution\HighResClock.h"
#include "spdlog\spdlog.h"

#define TRACE 1
#define TRACE_PERFORMANCE

void logtest()
{
    auto logger = spdlog::get("file_logger");
    logger->info("Log from application");
}



int main(int argc, const char* argv[])
{
#ifdef DEBUG
    spdlog::set_level(spdlog::level::debug);
#elif TRACE
    spdlog::set_level(spdlog::level::trace);
#else
    spdlog::set_level(spdlog::level::info);
#endif

    
    auto file_logger = spdlog::rotating_logger_mt("file_logger",RW::CORE::Util::getexepath(), 1048576 * 5, 3);
    file_logger->debug("******************");
    file_logger->debug("*Applicationstart*");
    file_logger->debug("******************");
    RW::tenStatus status = RW::tenStatus::nenError;
#ifdef TRACE_PERFORMANCE
     RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
    RW::CORE::ModuleLoader ml(file_logger);

    /*Load Plugins*/
	QList<RW::CORE::AbstractModule const*> list;
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

        RW::CORE::Kernel kernel(&context, "MyKernel", NVX_KERNEL_TEST,4, list[4], file_logger);
        if (kernel.IsInitialized())
        {
            status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 0);
            status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 1);
            status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 2);
            status = kernelManager.AddParameterToKernel(&kernel, RW::CORE::tenDir::nenInput, 3);

            status = kernelManager.FinalizeKernel(&kernel);

            RW::CORE::Node node(&graph, &kernel, file_logger);
            if (node.IsInitialized())
            {
                //Test data
                RW::VG::tstMyInitialiseControlStruct initControll;
                RW::VG::tstMyControlStruct control;
                RW::VG::tstMyDeinitialiseControlStruct deinitControl;
                initControll.u8Test = 10;
                control.u8Test = 10;
                deinitControl.u8Test = 10;

                node.SetParameterByIndex(0, (void*)&kernel, sizeof(RW::CORE::Kernel), &context);
                node.SetParameterByIndex(1, (void*)&initControll, sizeof(RW::VG::stMyInitialiseControlStruct), &context);
                node.SetParameterByIndex(2, (void*)&control, sizeof(RW::VG::stMyControlStruct), &context);
                node.SetParameterByIndex(3, (void*)&deinitControl, sizeof(RW::VG::stMyDeinitialiseControlStruct), &context);
            }
        }
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
    return 0;
}