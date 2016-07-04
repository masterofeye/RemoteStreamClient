#include "VPL_Viewer.h"
#include "spdlog\spdlog.h"
#include "Pipeline.h"
#include <thread>

#include <QApplication>

#define TRACE 1
#define TRACE_PERFORMANCE

void logtest()
{
    auto logger = spdlog::get("file_logger");
    logger->info("Log from application");
}


int main(int argc, char* argv[])
{
    auto file_logger = spdlog::stdout_logger_mt("file_logger");
    //auto file_logger = spdlog::rotating_logger_mt("file_logger", (qApp->applicationDirPath() + "/logfile.log").toStdString(), 1048576 * 5, 3);
    file_logger->debug("******************");
    file_logger->debug("*Applicationstart*");
    file_logger->debug("******************");

    tstPipelineParams params;
    params.file_logger = file_logger;

    Pipeline pipe;
    pipe.RunPipeline(params);
    //std::thread thPipe(pipe.RunPipeline, params);
    QApplication app(argc, argv);
    QWidget player;
    //RW::VPL::VPL_Viewer player;
    player.show();
    //app.exec();
    //thPipe.join();

#ifdef DEBUG
    spdlog::set_level(spdlog::level::debug);
#elif TRACE
    spdlog::set_level(spdlog::level::trace);
#else
    spdlog::set_level(spdlog::level::info);
#endif


    //std::thread (pipe.RunPipeline, params).detach();
    //std::thread wc(pipeline, params);

    //thGV.join();
    //wc.join();

    //thApp.join();

    return 0;
}

