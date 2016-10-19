#include "Pipeline.hpp"

void CPipethread::start()
{
    emit started();
    QThread::start();
}

CPipeline::CPipeline(tstPipelineParams* params)
{
    m_params = params;
    return;
}

int CPipeline::RunPipeline()
{
    tstPipelineParams params = *m_params;
    auto file_logger = params.file_logger;

    cudaError err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        file_logger->error("cudaDeviceReset failed!");
        return -1;
    }

    char *testLeak = new char[1000000];
    testLeak = nullptr;

    QList<RW::CORE::AbstractModule *> list;
    try
    {
#ifdef TRACE_PERFORMANCE
        RW::CORE::HighResClock::time_point t1 = RW::CORE::HighResClock::now();
#endif
        RW::CORE::ModuleLoader ml(file_logger);

        /*Load Plugins*/

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
            int iParentIndex = -1;

#ifdef SERVER
            RW::VGR::SIMU::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct;
            {
                videoGrabberInitialiseControlStruct.nFPS = 30;
                videoGrabberInitialiseControlStruct.nFrameHeight = 720;
                videoGrabberInitialiseControlStruct.nFrameWidth = 1920;
				//videoGrabberInitialiseControlStruct.nFrameHeight = 480;
				//videoGrabberInitialiseControlStruct.nFrameWidth = 640;
				videoGrabberInitialiseControlStruct.nNumberOfFrames = 345;
                videoGrabberInitialiseControlStruct.sFileName = "e:\\Video\\BR213_24bbp_5.avi";
            }
            RW::VGR::SIMU::tstVideoGrabberControlStruct videoGrabberControlStruct;
            {
                videoGrabberControlStruct.nCurrentFrameNumber = 0;
                videoGrabberControlStruct.nCurrentPositionMSec = 0;
            }
            RW::VGR::SIMU::tstVideoGrabberDeinitialiseControlStruct videoGrabberDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &videoGrabberInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::VGR::SIMU::tstVideoGrabberInitialiseControlStruct),
                &videoGrabberControlStruct,
                sizeof(RW::VGR::SIMU::tstVideoGrabberControlStruct),
                &videoGrabberDeinitialiseControlStruct,
                sizeof(RW::VGR::SIMU::tstVideoGrabberDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenVideoGrabber_SIMU) != RW::tenStatus::nenSuccess)
                file_logger->error("nenVideoGrabber_SIMU couldn't build correct");

#ifdef IMP_CROP
            RW::IMP::CROP::tstMyInitialiseControlStruct impCropInitialiseControlStruct;
            {

				// webcam
				//cv::Rect rect1(0, 0, 200, 250);
				//impCropInitialiseControlStruct.vFrameRect.push_back(rect1);
				// cuda out of memory?!

				cv::Rect rect1(0, 0, 200, 250);
				impCropInitialiseControlStruct.vFrameRect.push_back(rect1);
                cv::Rect rect2(1000, 350, 50, 50);
                impCropInitialiseControlStruct.vFrameRect.push_back(rect2);
                cv::Rect rect3(500, 100, 500, 500);
                impCropInitialiseControlStruct.vFrameRect.push_back(rect3);
            }

            RW::IMP::CROP::tstMyControlStruct impCropControlStruct;
            RW::IMP::CROP::tstMyDeinitialiseControlStruct impCropDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &impCropInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::IMP::CROP::tstMyInitialiseControlStruct),
                &impCropControlStruct,
                sizeof(RW::IMP::CROP::tstMyControlStruct),
                &impCropDeinitialiseControlStruct,
                sizeof(RW::IMP::CROP::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenGraphic_Crop) != RW::tenStatus::nenSuccess)
                file_logger->error("nenGraphic_Crop couldn't build correct");

#ifdef IMP_MERGE
            RW::IMP::MERGE::tstMyInitialiseControlStruct impMergeInitialiseControlStruct;
            RW::IMP::MERGE::tstMyControlStruct impMergeControlStruct;
            RW::IMP::MERGE::tstMyDeinitialiseControlStruct impMergeDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &impMergeInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::IMP::MERGE::tstMyInitialiseControlStruct),
                &impMergeControlStruct,
                sizeof(RW::IMP::MERGE::tstMyControlStruct),
                &impMergeDeinitialiseControlStruct,
                sizeof(RW::IMP::MERGE::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenGraphic_Merge) != RW::tenStatus::nenSuccess)
                file_logger->error("nenGraphic_Merge couldn't build correct");
#endif
#endif

            RW::IMP::COLOR_BGRTOYUV::tstMyInitialiseControlStruct impColorInitialiseControlStruct;
            RW::IMP::COLOR_BGRTOYUV::tstMyControlStruct impColorControlStruct;
            RW::IMP::COLOR_BGRTOYUV::tstMyDeinitialiseControlStruct impColorDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &impColorInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::IMP::COLOR_BGRTOYUV::tstMyInitialiseControlStruct),
                &impColorControlStruct,
                sizeof(RW::IMP::COLOR_BGRTOYUV::tstMyControlStruct),
                &impColorDeinitialiseControlStruct,
                sizeof(RW::IMP::COLOR_BGRTOYUV::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenGraphic_ColorBGRToYUV) != RW::tenStatus::nenSuccess)
                file_logger->error("nenGraphic_ColorBGRToYUV couldn't build correct");

            long lHeight = 0, lWidth = 0;
            {
#ifdef IMP_CROP
                long count = 0;
                while (count < impCropInitialiseControlStruct.vFrameRect.size())
                {
                    lWidth += impCropInitialiseControlStruct.vFrameRect[count].width;
                    lHeight = MAX(lHeight, impCropInitialiseControlStruct.vFrameRect[count].height);
                    count++;
                }
#else
                lWidth = videoGrabberInitialiseControlStruct.nFrameWidth;
                lHeight = videoGrabberInitialiseControlStruct.nFrameHeight;
#endif
            }

#ifdef ENC_INTEL
            RW::ENC::INTEL::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
            {
                encodeInitialiseControlStruct.pParams = new RW::ENC::INTEL::sInputParams();
                encodeInitialiseControlStruct.pParams->nWidth = lWidth;
                encodeInitialiseControlStruct.pParams->nHeight = lHeight;
				encodeInitialiseControlStruct.pParams->bUseHWLib = false;
                //encodeInitialiseControlStruct.pParams->memType = RW::ENC::INTEL::D3D9_MEMORY;
            }
            RW::ENC::INTEL::tstMyControlStruct encodeControlStruct;
            RW::ENC::INTEL::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &encodeInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::ENC::INTEL::tstMyInitialiseControlStruct),
                &encodeControlStruct,
                sizeof(RW::ENC::INTEL::tstMyControlStruct),
                &encodeDeinitialiseControlStruct,
                sizeof(RW::ENC::INTEL::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenEncode_INTEL) != RW::tenStatus::nenSuccess)
                file_logger->error("nenEncode_INTEL couldn't build correct");
#endif
#ifdef ENC_NVENC
            RW::ENC::NVENC::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
            {
                encodeInitialiseControlStruct.pParams = new RW::ENC::NVENC::EncodeConfig();
                encodeInitialiseControlStruct.pParams->nWidth += lWidth;
                encodeInitialiseControlStruct.pParams->nHeight = lHeight;//(encodeInitialiseControlStruct.pParams->nHeight > videoGrabberInitialiseControlStruct.nFrameHeight) ? encodeInitialiseControlStruct.pParams->nHeight : videoGrabberInitialiseControlStruct.nFrameHeight;
                encodeInitialiseControlStruct.pParams->fps = videoGrabberInitialiseControlStruct.nFPS;
                encodeInitialiseControlStruct.pParams->uBitstreamBufferSize = 2 * 1024 * 1024;
            }
            RW::ENC::NVENC::tstMyControlStruct encodeControlStruct;
            RW::ENC::NVENC::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &encodeInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::ENC::NVENC::tstMyInitialiseControlStruct),
                &encodeControlStruct,
                sizeof(RW::ENC::NVENC::tstMyControlStruct),
                &encodeDeinitialiseControlStruct,
                sizeof(RW::ENC::NVENC::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenEncode_NVIDIA) != RW::tenStatus::nenSuccess)
                file_logger->error("nenEncode_NVIDIA couldn't build correct");
#endif

            RW::SSR::LIVE555::tstMyInitialiseControlStruct streamInitialiseControlStruct;
            RW::SSR::LIVE555::tstMyControlStruct streamControlStruct;
            RW::SSR::LIVE555::tstMyDeinitialiseControlStruct streamDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &streamInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::SSR::LIVE555::tstMyInitialiseControlStruct),
                &streamControlStruct,
                sizeof(RW::SSR::LIVE555::tstMyControlStruct),
                &streamDeinitialiseControlStruct,
                sizeof(RW::SSR::LIVE555::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenStream_Simple) != RW::tenStatus::nenSuccess)
                file_logger->error("nenStream_Simple couldn't build correct");
#endif

#ifdef CLIENT
            RW::SCL::LIVE555::tstMyInitialiseControlStruct receiveInitCtrl;
            RW::SCL::LIVE555::tstMyControlStruct receiveCtrl;
            RW::SCL::LIVE555::tstMyDeinitialiseControlStruct receiveDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &receiveInitCtrl,
                iParentIndex++,
                sizeof(RW::SCL::LIVE555::tstMyInitialiseControlStruct),
                &receiveCtrl,
                sizeof(RW::SCL::LIVE555::tstMyControlStruct),
                &receiveDeinitCtrl,
                sizeof(RW::SCL::LIVE555::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenReceive_Simple) != RW::tenStatus::nenSuccess)
                file_logger->error("nenReceive_Simple couldn't build correct");

#ifdef DEC_INTEL
            // ---- If you use DEC\Intel set qViewer.setImgType(QImage::Format::Format_RGBX8888) in Main ----
            RW::DEC::INTEL::tstMyInitialiseControlStruct decodeInitCtrl;
            {
                decodeInitCtrl.inputParams = new RW::DEC::INTEL::tstInputParams();

                // Height and Width need to be transported via Connector configuration

				decodeInitCtrl.inputParams->nWidth = 1920;
				decodeInitCtrl.inputParams->nHeight = 720;
				//decodeInitCtrl.inputParams->nWidth = 640;
				//decodeInitCtrl.inputParams->nHeight = 480;
				decodeInitCtrl.inputParams->bCalLat = false;
                decodeInitCtrl.inputParams->bLowLat = false;
                decodeInitCtrl.inputParams->bUseHWLib = false;
                //decodeInitCtrl.inputParams->memType = RW::DEC::INTEL::D3D9_MEMORY;

                // ---- Do not use MFX_FOURCC_NV12 and IMP::COLOR_NV12TORGB. It does not work properly now. Use MFX_FOURCC_RGB4 instead ----
                decodeInitCtrl.inputParams->fourcc = MFX_FOURCC_RGB4;  
            }
            RW::DEC::INTEL::tstMyControlStruct decodeCtrl;
            RW::DEC::INTEL::tstMyDeinitialiseControlStruct decodeDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &decodeInitCtrl,
                iParentIndex++,
                sizeof(RW::DEC::INTEL::tstMyInitialiseControlStruct),
                &decodeCtrl,
                sizeof(RW::DEC::INTEL::tstMyControlStruct),
                &decodeDeinitCtrl,
                sizeof(RW::DEC::INTEL::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenDecoder_INTEL) != RW::tenStatus::nenSuccess)
                file_logger->error("nenDecoder_INTEL couldn't build correct");
#endif
#ifdef DEC_NVENC
            //---- If you use DEC\NVENC set qViewer.setImgType(QImage::Format::Format_RGB888) in Main ----
            RW::DEC::NVENC::tstMyInitialiseControlStruct decodeInitCtrl;
            {
                decodeInitCtrl.inputParams = new RW::DEC::NVENC::tstInputParams();

                // Height and Width need to be transported via Connector configuration

                decodeInitCtrl.inputParams->nHeight = 720;
                //    videoGrabberInitialiseControlStruct.nFrameHeight;
                decodeInitCtrl.inputParams->nWidth = 1920;
                //    videoGrabberInitialiseControlStruct.nFrameWidth;
            }
            RW::DEC::NVENC::tstMyControlStruct decodeCtrl;
            RW::DEC::NVENC::tstMyDeinitialiseControlStruct decodeDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &decodeInitCtrl,
                iParentIndex++,
                sizeof(RW::DEC::NVENC::tstMyInitialiseControlStruct),
                &decodeCtrl,
                sizeof(RW::DEC::NVENC::tstMyControlStruct),
                &decodeDeinitCtrl,
                sizeof(RW::DEC::NVENC::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenDecoder_NVIDIA) != RW::tenStatus::nenSuccess)
                file_logger->error("nenDecoder_NVIDIA couldn't build correct");


            //---- Use COLOR_NV12TORGB to convert the DEC\NVENC output into RGB ----
            RW::IMP::COLOR_NV12TORGB::tstMyInitialiseControlStruct impColor420InitialiseControlStruct;
            {
                impColor420InitialiseControlStruct.nHeight = decodeInitCtrl.inputParams->nHeight;
                impColor420InitialiseControlStruct.nWidth = decodeInitCtrl.inputParams->nWidth;
            }
            RW::IMP::COLOR_NV12TORGB::tstMyControlStruct impColor420ControlStruct;
            RW::IMP::COLOR_NV12TORGB::tstMyDeinitialiseControlStruct impColor420DeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &impColor420InitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::IMP::COLOR_NV12TORGB::tstMyInitialiseControlStruct),
                &impColor420ControlStruct,
                sizeof(RW::IMP::COLOR_NV12TORGB::tstMyControlStruct),
                &impColor420DeinitialiseControlStruct,
                sizeof(RW::IMP::COLOR_NV12TORGB::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenGraphic_ColorNV12ToRGB) != RW::tenStatus::nenSuccess)
                file_logger->error("nenGraphic_ColorYUV420ToRGB couldn't build correct");
#endif

            RW::VPL::QT_SIMPLE::tstMyInitialiseControlStruct playerInitCtrl;
            {
                playerInitCtrl.pViewer = params.pViewer;
            }
            RW::VPL::QT_SIMPLE::tstMyControlStruct playerCtrl;
            RW::VPL::QT_SIMPLE::tstMyDeinitialiseControlStruct playerDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &playerInitCtrl,
                iParentIndex++,
                sizeof(RW::VPL::QT_SIMPLE::tstMyInitialiseControlStruct),
                &playerCtrl,
                sizeof(RW::VPL::QT_SIMPLE::tstMyControlStruct),
                &playerDeinitCtrl,
                sizeof(RW::VPL::QT_SIMPLE::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenPlayback_Simple) != RW::tenStatus::nenSuccess)
                file_logger->error("nenPlayback_Simple couldn't build correct");
#endif
            uint32_t count = 0;
            RW::tenStatus res = RW::tenStatus::nenSuccess;
            if (graph.VerifyGraph() == RW::tenStatus::nenSuccess)
            {
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point  tAfterInit = RW::CORE::HighResClock::now();
#endif
                for (long lIndex = 0; lIndex < 50; lIndex++)
                {
                    if (res == RW::tenStatus::nenSuccess)
                    {

                        file_logger->debug("*********************************");
                        file_logger->debug("*Graph excecution for Frame #") << lIndex;
                        file_logger->debug("*********************************");
                    }
                    res = graph.ScheduleGraph();
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Prepare Graph: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << " ms.";
                    t1 = RW::CORE::HighResClock::now();
#endif
                    graph.WaitGraph();
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Graph execution: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << " ms.";

                    MEMORYSTATUSEX statex;
                    statex.dwLength = sizeof(statex);
                    GlobalMemoryStatusEx(&statex);
                    long lDIV = 1048576;
                    file_logger->trace() << "There are Memory load Mbytes: " << statex.dwMemoryLoad << "%";
                    file_logger->trace() << "There are total Mbytes of physical memory: " << statex.ullTotalPhys / lDIV;
                    file_logger->trace() << "There are total Mbytes of virtual memory: " << statex.ullTotalVirtual / lDIV;
                    file_logger->trace() << "There are available Mbytes of physical memory: " << statex.ullAvailPhys / lDIV;
                    file_logger->trace() << "There are available Mbytes of virtual memory: " << statex.ullAvailVirtual / lDIV;
                    file_logger->trace() << "There are occupied Mbytes of physical memory: " << (statex.ullTotalPhys - statex.ullAvailPhys) / lDIV;
                    file_logger->trace() << "There are occupied Mbytes of virtual memory: " << (statex.ullTotalVirtual - statex.ullAvailVirtual) / lDIV;

#endif
                    count++;
                }
#ifdef TRACE_PERFORMANCE
                RW::CORE::HighResClock::time_point tAfterAllDoRenders = RW::CORE::HighResClock::now();
                file_logger->trace() << "DoRender for " << count << " frames took " << RW::CORE::HighResClock::diffMilli(tAfterInit, tAfterAllDoRenders).count() << "ms.";
                file_logger->trace() << "This is an average fps of " << (float)count / ((float)RW::CORE::HighResClock::diffMilli(tAfterInit, tAfterAllDoRenders).count() / 1000.0);
                t1 = RW::CORE::HighResClock::now();
#endif
            }

            /*******Cleanup. Whatever has been created here has to be destroyed here. Modules do not do that. ******/
#ifdef SERVER
            //SAFE_DELETE(encodeInitialiseControlStruct.pstEncodeConfig);
            //SAFE_DELETE(encodeInitialiseControlStruct.pParams);
#endif
#ifdef CLIENT
            SAFE_DELETE(decodeInitCtrl.inputParams);
#endif
            cudaError err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("Pipeline: Device synchronize failed! Error = %d\n", err);
                return -1;
            }

        }
    }
    catch (...)
    {
        file_logger->flush();
        return -1;
    }

    //Delete all modules;
    for (auto var : list)
    {
        if (var){
            delete var;
        }
    }


    return 0;
}
