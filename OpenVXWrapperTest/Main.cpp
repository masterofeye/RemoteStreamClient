#include "Pipeline.hpp"
#include <io.h>
#include <fcntl.h>

int main(int argc, char* argv[])
{
	_setmode(_fileno(stdout), _O_BINARY);
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
//#ifdef RS_CLIENT
//    RW::VPL::QT_SIMPLE::VPL_Viewer qViewer;
//	qViewer.setParams(1920, 720);
//	//qViewer.setParams(640, 480);
//    QImage::Format format;
//#ifdef DEC_INTEL
//    format = QImage::Format::Format_RGBX8888;
//#endif
//#ifdef DEC_NVENC
//    format = QImage::Format::Format_RGB888;
//#endif
//    qViewer.setImgType(format);
//    qViewer.resize(1924, 724);
//    qViewer.show();
//    params.pViewer = &qViewer;
//#endif

    


#ifdef RS_SERVER  // for RS_CLIENT this is being handled in RSAPP
    CPipeline pipe(&params);
    CPipethread thread;

    QObject::connect(&thread, SIGNAL(started()), &pipe, SLOT(RunPipeline()), Qt::DirectConnection);
    pipe.moveToThread(&thread);
    thread.start();
#endif
    app.exec();
#ifdef RS_SERVER
    thread.wait();
#endif
    return 0;
}

