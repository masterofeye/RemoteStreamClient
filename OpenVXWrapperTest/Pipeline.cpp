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

            RW::VG::tstVideoGrabberInitialiseControlStruct videoGrabberInitialiseControlStruct;
            {
                videoGrabberInitialiseControlStruct.nFPS = 30;
                videoGrabberInitialiseControlStruct.nFrameHeight = 720;
                videoGrabberInitialiseControlStruct.nFrameWidth = 1920;
                videoGrabberInitialiseControlStruct.nNumberOfFrames = 345;
                videoGrabberInitialiseControlStruct.sFileName = "C:\\Projekte\\BR213_24bbp_5.avi";
            }
            RW::VG::tstVideoGrabberControlStruct videoGrabberControlStruct;
            {
                videoGrabberControlStruct.Output = new cv::cuda::GpuMat();
                //Will be filled by the VideoGrabber itself
                //videoGrabberControlStruct.pOutputData->pBuffer = nullptr;
                videoGrabberControlStruct.nCurrentFrameNumber = 0;
                videoGrabberControlStruct.nCurrentPositionMSec = 0;
            }
            RW::VG::tstVideoGrabberDeinitialiseControlStruct videoGrabberDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &videoGrabberInitialiseControlStruct,
                iParentIndex++,
                sizeof(videoGrabberInitialiseControlStruct),
                &videoGrabberControlStruct,
                sizeof(RW::VG::tstVideoGrabberControlStruct),
                &videoGrabberDeinitialiseControlStruct,
                sizeof(RW::VG::tstVideoGrabberDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenVideoGrabber_SIMU) != RW::tenStatus::nenSuccess)
                file_logger->error("nenVideoGrabber_SIMU couldn't build correct");

            //std::vector<cv::Rect> vRect;
            //vRect.push_back(cv::Rect(100, 100, 50, 50)); 
            //vRect.push_back(cv::Rect(50, 50, 100, 100)); 
            //         RW::IMP::CROP::tstMyInitialiseControlStruct impCropInitialiseControlStruct;
            //         {
            //	impCropInitialiseControlStruct.vFrameRect = vRect;
            //}
            //RW::IMP::CROP::tstMyControlStruct impCropControlStruct;
            //         {
            //	impCropControlStruct.pInput = new RW::IMP::cInputBase(
            //		RW::IMP::cInputBase::tstImportImg{
            //		videoGrabberInitialiseControlStruct.nFrameWidth,
            //		videoGrabberInitialiseControlStruct.nFrameHeight,
            //		videoGrabberControlStruct.pOutputData->pBuffer }, true);
            //	impCropControlStruct.pvOutput = new std::vector<cv::cuda::GpuMat*>;
            //	impCropControlStruct.pvOutput->push_back(new cv::cuda::GpuMat);
            //	impCropControlStruct.pvOutput->push_back(new cv::cuda::GpuMat);
            //}
            //         RW::IMP::CROP::tstMyDeinitialiseControlStruct impCropDeinitialiseControlStruct;

            //        if (builder.BuildNode(&kernelManager,
            //        	&impCropInitialiseControlStruct,
            //iParentIndex++,
            //sizeof(RW::IMP::CROP::tstMyInitialiseControlStruct),
            //        	&impCropControlStruct,
            //sizeof(RW::IMP::CROP::tstMyControlStruct),
            //        	&impCropDeinitialiseControlStruct,
            //sizeof(RW::IMP::CROP::tstMyDeinitialiseControlStruct),
            //        	RW::CORE::tenSubModule::nenGraphic_Crop) != RW::tenStatus::nenSuccess)
            //        	file_logger->error("nenGraphic_Crop couldn't build correct");

            //RW::IMP::MERGE::tstMyInitialiseControlStruct impMergeInitialiseControlStruct;
            //RW::IMP::MERGE::tstMyControlStruct impMergeControlStruct;
            //{
            //	impMergeControlStruct.pvInput = new std::vector<RW::IMP::cInputBase*>;
            //	impMergeControlStruct.pOutput = impCropControlStruct.pvOutput->at(0);
            //	for (int iIndex = 0; iIndex < impCropControlStruct.pvOutput->size(); iIndex++)
            //	{
            //		impMergeControlStruct.pvInput->push_back(new RW::IMP::cInputBase(impCropControlStruct.pvOutput->at(iIndex)));
            //	}
            //}
            //RW::IMP::MERGE::tstMyDeinitialiseControlStruct impMergeDeinitialiseControlStruct;

            //if (builder.BuildNode(&kernelManager,
            //	&impMergeInitialiseControlStruct,
            //	iParentIndex++,
            //	sizeof(RW::IMP::MERGE::tstMyInitialiseControlStruct),
            //	&impMergeControlStruct,
            //	sizeof(RW::IMP::MERGE::tstMyControlStruct),
            //	&impMergeDeinitialiseControlStruct,
            //	sizeof(RW::IMP::MERGE::tstMyDeinitialiseControlStruct),
            //	RW::CORE::tenSubModule::nenGraphic_Merge) != RW::tenStatus::nenSuccess)
            //	file_logger->error("nenGraphic_Color couldn't build correct");

            RW::IMP::COLOR::tstMyInitialiseControlStruct impColorInitialiseControlStruct;
            CUdeviceptr arrYUV;
            RW::IMP::COLOR::tstMyControlStruct impColorControlStruct;
            {
                impColorControlStruct.pInput = new RW::IMP::cInputBase(videoGrabberControlStruct.Output);
                size_t pitch;
                cudaError err;
                err = cudaMallocPitch((void**)&arrYUV, &pitch, videoGrabberInitialiseControlStruct.nFrameWidth, videoGrabberInitialiseControlStruct.nFrameHeight * 3 / 2);
                impColorControlStruct.pOutput = new RW::IMP::cOutputBase(arrYUV, true);
            }
            RW::IMP::COLOR::tstMyDeinitialiseControlStruct impColorDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &impColorInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::IMP::COLOR::tstMyInitialiseControlStruct),
                &impColorControlStruct,
                sizeof(RW::IMP::COLOR::tstMyControlStruct),
                &impColorDeinitialiseControlStruct,
                sizeof(RW::IMP::COLOR::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenGraphic_Color) != RW::tenStatus::nenSuccess)
                file_logger->error("nenGraphic_Color couldn't build correct");

            RW::ENC::tstMyInitialiseControlStruct encodeInitialiseControlStruct;
            {
                encodeInitialiseControlStruct.pstEncodeConfig = new RW::ENC::EncodeConfig();
                encodeInitialiseControlStruct.pstEncodeConfig->width += videoGrabberInitialiseControlStruct.nFrameWidth;
                encodeInitialiseControlStruct.pstEncodeConfig->height = (encodeInitialiseControlStruct.pstEncodeConfig->height > videoGrabberInitialiseControlStruct.nFrameHeight) ? encodeInitialiseControlStruct.pstEncodeConfig->height : videoGrabberInitialiseControlStruct.nFrameHeight;
                encodeInitialiseControlStruct.pstEncodeConfig->fps = videoGrabberInitialiseControlStruct.nFPS;
                encodeInitialiseControlStruct.pstEncodeConfig->uBitstreamBufferSize = 2 * 1024 * 1024;
            }
            RW::ENC::tstMyControlStruct encodeControlStruct;
            {
                encodeControlStruct.pcuYUVArray = impColorControlStruct.pOutput->_pcuYUV420;
                encodeControlStruct.pPayload = new RW::tstBitStream();
                tstPayloadMsg Msg;
                Msg.u32Timestamp = videoGrabberControlStruct.nCurrentPositionMSec;
                Msg.u32FrameNbr = videoGrabberControlStruct.nCurrentFrameNumber;
                encodeControlStruct.pPayload->u32Size = sizeof(stPayloadMsg);
                encodeControlStruct.pPayload->pBuffer = (uint8_t*)&Msg;
                encodeControlStruct.pstBitStream = new RW::tstBitStream;
            }
            RW::ENC::tstMyDeinitialiseControlStruct encodeDeinitialiseControlStruct;

            if (builder.BuildNode(&kernelManager,
                &encodeInitialiseControlStruct,
                iParentIndex++,
                sizeof(RW::ENC::tstMyInitialiseControlStruct),
                &encodeControlStruct,
                sizeof(RW::ENC::tstMyControlStruct),
                &encodeDeinitialiseControlStruct,
                sizeof(RW::ENC::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenEncode_NVIDIA) != RW::tenStatus::nenSuccess)
                file_logger->error("nenEncode_NVIDIA couldn't build correct");

            //FILE *pFile;
            //pFile = fopen("C:\\dummy\\HeavyHand.264", "rb");
            //if (!pFile)
            //{
            //    file_logger->error("File did not load!");
            //    return -1;
            //}
            //// obtain file size:
            //fseek(pFile, 0, SEEK_END);
            //long lSize = ftell(pFile);
            //rewind(pFile);

            //// allocate memory to contain the whole file:
            //uint8_t *buffer = (uint8_t*)malloc(sizeof(uint8_t)*lSize);
            //// copy the file into the buffer:
            //size_t result = fread(buffer, sizeof(uint8_t), lSize, pFile);
            //if (!buffer)
            //{ 
            //    file_logger->error("Empty buffer!");
            //    return -1;
            //}
            //fclose(pFile);

            //RW::tstBitStream *pBitStream = new RW::tstBitStream();
            //pBitStream->pBuffer = buffer;
            //pBitStream->u32Size = lSize;

            //RW::tstBitStream *pPayload = new RW::tstBitStream();
            //pPayload->pBuffer = nullptr;
            //pPayload->u32Size = sizeof(stPayloadMsg);

            RW::DEC::tstMyInitialiseControlStruct decodeInitCtrl;
            {
                decodeInitCtrl.inputParams = new RW::DEC::tstInputParams();
                decodeInitCtrl.inputParams->Height = //1080;
                    videoGrabberInitialiseControlStruct.nFrameHeight;
                decodeInitCtrl.inputParams->Width = //1920;
                    videoGrabberInitialiseControlStruct.nFrameWidth;
                decodeInitCtrl.inputParams->fourcc = MFX_FOURCC_RGB4;
                decodeInitCtrl.inputParams->memType = RW::DEC::D3D9_MEMORY;
            }
            RW::DEC::tstMyControlStruct decodeCtrl;
            {
                decodeCtrl.pstEncodedStream = //pBitStream;
                    encodeControlStruct.pstBitStream;
                decodeCtrl.pPayload = //pPayload;
                    encodeControlStruct.pPayload;
                decodeCtrl.pOutput = new RW::tstBitStream();
                uint32_t size = videoGrabberInitialiseControlStruct.nFrameWidth * videoGrabberInitialiseControlStruct.nFrameHeight * sizeof(uint8_t) * 4;
                decodeCtrl.pOutput->pBuffer = new uint8_t[size];
                decodeCtrl.pOutput->u32Size = 0;
            }
            RW::DEC::tstMyDeinitialiseControlStruct decodeDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &decodeInitCtrl,
                iParentIndex++,
                sizeof(RW::DEC::tstMyInitialiseControlStruct),
                &decodeCtrl,
                sizeof(RW::DEC::tstMyControlStruct),
                &decodeDeinitCtrl,
                sizeof(RW::DEC::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenDecoder_INTEL) != RW::tenStatus::nenSuccess)
                file_logger->error("nenDecoder_INTEL couldn't build correct");

            RW::VPL::tstMyInitialiseControlStruct playerInitCtrl;
            {
                playerInitCtrl.pViewer = params.pViewer;
            }
            RW::VPL::tstMyControlStruct playerCtrl;
            {
                playerCtrl.pstBitStream = decodeCtrl.pOutput;
                playerCtrl.TimeStamp = //1000;
                    videoGrabberControlStruct.nCurrentPositionMSec;
            }
            RW::VPL::tstMyDeinitialiseControlStruct playerDeinitCtrl;

            if (builder.BuildNode(&kernelManager,
                &playerInitCtrl,
                iParentIndex++,
                sizeof(RW::VPL::tstMyInitialiseControlStruct),
                &playerCtrl,
                sizeof(RW::VPL::tstMyControlStruct),
                &playerDeinitCtrl,
                sizeof(RW::VPL::tstMyDeinitialiseControlStruct),
                RW::CORE::tenSubModule::nenPlayback_Simple) != RW::tenStatus::nenSuccess)
                file_logger->error("nenPlayback_Simple couldn't build correct");

            uint32_t count = 0;
            RW::tenStatus res = RW::tenStatus::nenSuccess;
            if (graph.VerifyGraph() == RW::tenStatus::nenSuccess)
            {
                for (uint16_t u16Index = 0; u16Index < 100; u16Index++)
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
                    file_logger->trace() << "Prepare Graph: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
                    t1 = RW::CORE::HighResClock::now();
#endif
                    graph.WaitGraph();
#ifdef TRACE_PERFORMANCE
                    t2 = RW::CORE::HighResClock::now();
                    file_logger->trace() << "Graph execution: " << RW::CORE::HighResClock::diffMilli(t1, t2).count() << "ms.";
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

            //SAFE_DELETE(videoGrabberControlStruct.pOutputData->pBuffer);
            SAFE_DELETE(videoGrabberControlStruct.Output);

            //for (int iIndex = 0; iIndex < impCropControlStruct.pvOutput->size(); iIndex++)
            //{
            //	SAFE_DELETE(impCropControlStruct.pvOutput->at(iIndex));
            //	SAFE_DELETE(impMergeControlStruct.pvInput->at(iIndex));
            //}
            //SAFE_DELETE(impCropControlStruct.pvOutput);
            //SAFE_DELETE(impMergeControlStruct.pvInput);

            SAFE_DELETE(impColorControlStruct.pInput);
            SAFE_DELETE(impColorControlStruct.pOutput);

            SAFE_DELETE(encodeInitialiseControlStruct.pstEncodeConfig);
            SAFE_DELETE(encodeControlStruct.pPayload);

            // terminate testing data
            //free(buffer);
            //SAFE_DELETE(pPayload);
            //SAFE_DELETE(pBitStream);
            //SAFE_DELETE(pFile);

            SAFE_DELETE(decodeInitCtrl.inputParams);
            SAFE_DELETE(decodeCtrl.pOutput->pBuffer);
            SAFE_DELETE(decodeCtrl.pOutput);

            cudaError err = cudaDeviceSynchronize();
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
