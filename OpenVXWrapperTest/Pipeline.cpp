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

            //// ---- If you use DEC\Intel set qViewer.setImgType(QImage::Format::Format_RGBX8888) in Main ----
            //RW::DEC::INTEL::tstMyInitialiseControlStruct decodeInitCtrl;
            //{
            //    decodeInitCtrl.inputParams = new RW::DEC::INTEL::tstInputParams();

            //    // Height and Width need to be transported via Connector configuration

            //    decodeInitCtrl.inputParams->Height = 720;
            //    decodeInitCtrl.inputParams->Width = 1920;
            //    decodeInitCtrl.inputParams->bCalLat = false;
            //    decodeInitCtrl.inputParams->bLowLat = false;
            //    decodeInitCtrl.inputParams->bUseHWLib = false;
            //    // ---- Do not use MFX_FOURCC_NV12 and COLOR_NV12TORGB. It does not work properly now. Use MFX_FOURCC_RGB4 instead ----
            //    decodeInitCtrl.inputParams->fourcc = MFX_FOURCC_RGB4;  
            //}
            //RW::DEC::INTEL::tstMyControlStruct decodeCtrl;
            //RW::DEC::INTEL::tstMyDeinitialiseControlStruct decodeDeinitCtrl;

            //if (builder.BuildNode(&kernelManager,
            //    &decodeInitCtrl,
            //    iParentIndex++,
            //    sizeof(RW::DEC::INTEL::tstMyInitialiseControlStruct),
            //    &decodeCtrl,
            //    sizeof(RW::DEC::INTEL::tstMyControlStruct),
            //    &decodeDeinitCtrl,
            //    sizeof(RW::DEC::INTEL::tstMyDeinitialiseControlStruct),
            //    RW::CORE::tenSubModule::nenDecoder_INTEL) != RW::tenStatus::nenSuccess)
            //    file_logger->error("nenDecoder_INTEL couldn't build correct");

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

            uint32_t count = 0;
            RW::tenStatus res = RW::tenStatus::nenSuccess;
            if (graph.VerifyGraph() == RW::tenStatus::nenSuccess)
            {
                for (uint16_t u16Index = 0; u16Index < 464; u16Index++)
                {
                    res = graph.ScheduleGraph();
                    if (res == RW::tenStatus::nenSuccess)
                    {

                        file_logger->debug("******************");
                        file_logger->debug("*Graph excecution*");
                        file_logger->debug("******************");
                    }
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Prepare Graph: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << " ms.";
                    t1 = RW::CORE::HighResClock::now();
#endif
                    graph.WaitGraph();
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Graph execution: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << " ms.";
                    file_logger->trace() << "Process Frame #" << count;
#endif
                    count++;
                }
            }

            /*********Check if payload is correctly*******/
            //tstPayloadMsg *pMsg = (tstPayloadMsg*)(decodeCtrl.pPayload->pBuffer);
            //uint32_t u32FrameNbr = pMsg->u32FrameNbr;
            //uint32_t u32Timestamp = pMsg->u32Timestamp;
            /*********************************************/

            /*******Cleanup. Whatever has been created here has to be destroyed here. Modules do not do that. ******/

            SAFE_DELETE(decodeInitCtrl.inputParams);

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
