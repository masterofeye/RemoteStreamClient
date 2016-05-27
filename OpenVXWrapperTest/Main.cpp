#include <iostream>
#include <OpenVXWrapper.h>
#include "ModuleLoader.hpp"
/*Modules*/
#include "IMP_ConvColorFrames.hpp"
#include "Plugin1.hpp"
#include "GraphBuilder.h"
#include "ENC_CudaInterop.hpp"
#include "VideoGrabberSimu.hpp"

#include "HighResolution\HighResClock.h"
#include "spdlog\spdlog.h"

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
    QApplication app(argc, argv);

#ifdef DEBUG
        spdlog::set_level(spdlog::level::debug);
#elif TRACE
        spdlog::set_level(spdlog::level::trace);
#else
        spdlog::set_level(spdlog::level::info);
#endif

        auto file_logger = spdlog::stdout_logger_mt("file_logger");
        //auto file_logger = spdlog::rotating_logger_mt("file_logger", (qApp->applicationDirPath() + "/logfile.log").toStdString(), 1048576 * 5, 3);
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

                RW::VG::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct;
                videoGrabberInitialiseControlStruct.enColorSpace = RW::VG::nenRGB;
                videoGrabberInitialiseControlStruct.nFPS = 60;
                videoGrabberInitialiseControlStruct.nFrameHeight = 1920;
                videoGrabberInitialiseControlStruct.nFrameWidth = 720;
                videoGrabberInitialiseControlStruct.nNumberOfFrames = 1000;
                videoGrabberInitialiseControlStruct.sFileName = "c:\\BR213_24bbp_5.avi";

                RW::VG::tstVideoGrabberControlStruct videoGrabberControlStruct;
                videoGrabberControlStruct.nCurrentFrameNumber = 0;
                videoGrabberControlStruct.nCurrentPositionMSec = 0;
                videoGrabberControlStruct.nDataLength = 4147200;
                videoGrabberControlStruct.pData = new uint8_t[4147200];

                RW::VG::tstVideoGrabberDeinitialiseControlStruct videoGrabberDeinitialiseControlStruct;

                RW::IMP::tstMyInitialiseControlStruct impInitialiseControlStruct;
                //RW::IMP::tstRectStruct *pTest = new RW::IMP::tstRectStruct{ 0, 0, 100, 100 };
                //impInitialiseControlStruct.pstFrameRect = pTest; // only for nenGraphic_Crop
                RW::IMP::tstMyControlStruct impControlStruct;
                RW::IMP::tstInputParams *test = new RW::IMP::tstInputParams();
                impControlStruct.pcInput = new RW::IMP::cInputBase(test);
                cv::cuda::GpuMat *pgMat = new cv::cuda::GpuMat();
                impControlStruct.pcOutput = new RW::IMP::cOutputBase(pgMat);

                RW::IMP::tstMyDeinitialiseControlStruct impDeinitialiseControlStruct;

                RW::ENC::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
                RW::ENC::tstMyControlStruct encodeControlStruct;
                RW::ENC::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

                RW::CORE::Kernel *Kernel = nullptr;
                if (builder.BuildNode(&kernelManager,
                    &videoGrabberInitialiseControlStruct,
                    sizeof(videoGrabberInitialiseControlStruct),
                    &videoGrabberControlStruct,
                    sizeof(RW::VG::tstVideoGrabberControlStruct),
                    &videoGrabberDeinitialiseControlStruct,
                    sizeof(RW::VG::tstVideoGrabberDeinitialiseControlStruct), 
                    RW::CORE::tenSubModule::nenVideoGrabber_SIMU,
                    &Kernel) != RW::tenStatus::nenSuccess)
                    file_logger->error("nenVideoGrabber_SIMU couldn't build correct");
                
                //if (builder.BuildNode(&kernelManager, 
                //    &impInitialiseControlStruct, 
                //    sizeof(RW::IMP::tstMyInitialiseControlStruct), 
                //    &impControlStruct, 
                //    sizeof(RW::IMP::tstMyControlStruct), 
                //    &impDeinitialiseControlStruct, 
                //    sizeof(RW::IMP::tstMyDeinitialiseControlStruct), 
                //    RW::CORE::tenSubModule::nenGraphic_Color) != RW::tenStatus::nenSuccess)
                //    file_logger->error("nenGraphic_Color couldn't build correct");
                delete pgMat;
                delete impControlStruct.pcInput;
                delete impControlStruct.pcOutput;


                //if (builder.BuildNode(&kernelManager, 
                //    &encodeInitialiseControlStruct, 
                //    sizeof(RW::ENC::tstMyInitialiseControlStruct),
                //    &encodeControlStruct, 
                //    sizeof(RW::ENC::tstMyControlStruct),
                //    &encodeDeinitialiseControlStruct,
                //    sizeof(RW::ENC::tstMyDeinitialiseControlStruct),
                //    RW::CORE::tenSubModule::nenEncode_NVIDIA) != RW::tenStatus::nenSuccess)
                //    file_logger->error("nenEncode_NVIDIA couldn't build correct");

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

                        qDebug() << ((RW::VG::tstVideoGrabberControlStruct*)(Kernel->m_ControlStruct))->nCurrentFrameNumber;
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