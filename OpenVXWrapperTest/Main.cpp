#include "Pipeline.hpp"


int main(int argc, char* argv[])
{
    auto file_logger = spdlog::stdout_logger_mt("file_logger");
    //auto file_logger = spdlog::rotating_logger_mt("file_logger", (qApp->applicationDirPath() + "/logfile.log").toStdString(), 1048576 * 5, 3);
    file_logger->debug("******************");
    file_logger->debug("*Applicationstart*");
    file_logger->debug("******************");

#ifdef DEBUG
    spdlog::set_level(spdlog::level::debug);
#elif TRACE
    spdlog::set_level(spdlog::level::trace);
#else
    spdlog::set_level(spdlog::level::info);
#endif

    tstPipelineParams params;
    params.file_logger = file_logger;

    QApplication app(argc, argv);
    RW::VPL::QT_SIMPLE::VPL_Viewer qViewer;
    qViewer.setParams(1920, 720);
    qViewer.setImgType(QImage::Format::Format_RGBX8888);
    qViewer.resize(1924, 724);
    qViewer.show();
    params.pViewer = &qViewer;

    CPipeline pipe(&params);
    CPipethread thread;

    QObject::connect(&thread, SIGNAL(started()), &pipe, SLOT(RunPipeline()), Qt::AutoConnection);
    pipe.moveToThread(&thread);

    thread.start();

    app.exec();

    thread.wait();
    return 0;
}

